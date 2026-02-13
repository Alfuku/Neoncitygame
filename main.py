import os
import random
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship
from passlib.context import CryptContext

# ---------- CONFIG ----------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////tmp/city.db")

# ---------- DATABASE ----------
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# ---------- SECURITY ----------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)

# ---------- MODELS ----------
class PlayerDB(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)

    money = Column(Float, default=100.0)
    energy = Column(Integer, default=100)
    max_energy = Column(Integer, default=100)
    level = Column(Integer, default=1)
    experience = Column(Integer, default=0)

    strength = Column(Integer, default=1)
    agility = Column(Integer, default=1)
    intelligence = Column(Integer, default=1)
    charisma = Column(Integer, default=1)

    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)

    inventory = relationship("InventoryDB", back_populates="player", cascade="all, delete-orphan")


class InventoryDB(Base):
    __tablename__ = "inventory"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"))
    item_type = Column(String)
    item_id = Column(String)
    equipped = Column(Boolean, default=True)

    player = relationship("PlayerDB", back_populates="inventory")


Base.metadata.create_all(bind=engine)

# ---------- FASTAPI ----------
app = FastAPI(title="City of Syndicates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------- GAME DATA ----------
WEAPONS = {
    "brass_knuckles": {"name": "Brass Knuckles", "damage": 5, "price": 200, "level": 1},
    "pistol": {"name": "Pistol", "damage": 25, "price": 2000, "level": 5},
}

ARMOR = {
    "leather_jacket": {"name": "Leather Jacket", "defense": 5, "price": 300, "level": 1},
}

CRIMES = [
    {"name": "Pickpocket", "energy": 10, "reward": (20, 50), "exp": 5, "success": 0.9},
    {"name": "Bank Heist", "energy": 60, "reward": (500, 1000), "exp": 50, "success": 0.4},
]

# ---------- SCHEMAS ----------
class PlayerCreate(BaseModel):
    username: str
    password: str

class PlayerLogin(BaseModel):
    username: str
    password: str

class Action(BaseModel):
    username: str

class BuyItem(BaseModel):
    username: str
    item_id: str
    item_type: str

# ---------- HELPERS ----------
def get_player(username: str, db: Session):
    player = db.query(PlayerDB).filter(PlayerDB.username == username).first()
    if not player:
        raise HTTPException(404, "Player not found")
    return player

def add_exp(player: PlayerDB, amount: int):
    player.experience += amount
    new_level = 1 + (player.experience // 100)
    if new_level > player.level:
        player.level = new_level
        player.max_energy += 10

# ---------- ROUTES ----------
@app.get("/")
def root():
    return {"status": "City of Syndicates API LIVE"}

@app.post("/register")
def register(player: PlayerCreate, db: Session = Depends(get_db)):
    if db.query(PlayerDB).filter(PlayerDB.username == player.username).first():
        raise HTTPException(400, "Username taken")

    new_player = PlayerDB(
        username=player.username,
        password=hash_password(player.password)
    )
    db.add(new_player)
    db.commit()
    db.refresh(new_player)

    db.add(InventoryDB(player_id=new_player.id, item_type="weapon", item_id="brass_knuckles"))
    db.commit()

    return {"message": "Registered"}

@app.post("/login")
def login(player: PlayerLogin, db: Session = Depends(get_db)):
    user = get_player(player.username, db)
    if not verify_password(player.password, user.password):
        raise HTTPException(400, "Invalid credentials")
    return {"message": "Login successful", "username": user.username}

@app.post("/crime")
def crime(action: Action, db: Session = Depends(get_db)):
    user = get_player(action.username, db)
    crime = random.choice(CRIMES)

    if user.energy < crime["energy"]:
        raise HTTPException(400, "Not enough energy")

    user.energy -= crime["energy"]

    if random.random() <= crime["success"]:
        reward = random.randint(*crime["reward"])
        user.money += reward
        user.wins += 1
        add_exp(user, crime["exp"])
        db.commit()
        return {"result": "SUCCESS", "reward": reward}
    else:
        user.losses += 1
        db.commit()
        return {"result": "FAILED"}

@app.post("/rest")
def rest(action: Action, db: Session = Depends(get_db)):
    user = get_player(action.username, db)
    user.energy = min(user.max_energy, user.energy + 40)
    db.commit()
    return {"energy": user.energy}

@app.get("/armory")
def armory():
    return {"weapons": WEAPONS, "armor": ARMOR}

@app.post("/armory/buy")
def buy_item(data: BuyItem, db: Session = Depends(get_db)):
    player = get_player(data.username, db)

    item = WEAPONS.get(data.item_id) if data.item_type == "weapon" else ARMOR.get(data.item_id)
    if not item:
        raise HTTPException(404, "Item not found")

    if player.money < item["price"]:
        raise HTTPException(400, "Not enough money")

    player.money -= item["price"]

    # unequip previous same type
    db.query(InventoryDB).filter(
        InventoryDB.player_id == player.id,
        InventoryDB.item_type == data.item_type
    ).update({"equipped": False})

    db.add(InventoryDB(player_id=player.id, item_type=data.item_type, item_id=data.item_id, equipped=True))
    db.commit()

    return {"message": f"Bought {item['name']}"}
