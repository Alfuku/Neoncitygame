import os
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    ForeignKey, Boolean
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship

from passlib.context import CryptContext
from jose import jwt

# =========================
# CONFIG / DATABASE
# =========================
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in Render Environment Variables")

# Render sometimes gives postgres://
DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# =========================
# SECURITY
# =========================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


def make_token(username: str) -> str:
    payload = {"sub": username, "exp": datetime.utcnow() + timedelta(days=7)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


# =========================
# MODELS
# =========================
class PlayerDB(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)

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
    player_id = Column(Integer, ForeignKey("players.id"), index=True)

    item_type = Column(String(20), index=True)  # weapon / armor
    item_id = Column(String(100), index=True)
    equipped = Column(Boolean, default=False)

    player = relationship("PlayerDB", back_populates="inventory")


Base.metadata.create_all(bind=engine)
# TEMP: Reset tables (remove after first successful run)
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

# =========================
# GAME DATA
# =========================
WEAPONS = {
    "brass_knuckles": {"name": "Brass Knuckles", "damage": 5, "price": 80, "level": 1},
    "pistol": {"name": "Pistol", "damage": 25, "price": 2000, "level": 5},
}

ARMOR = {
    "leather_jacket": {"name": "Leather Jacket", "defense": 5, "price": 120, "level": 1},
}

CRIMES = [
    {"name": "Pickpocket", "energy": 10, "reward": (20, 50), "exp": 5, "success": 0.9},
    {"name": "Mugging", "energy": 20, "reward": (60, 140), "exp": 12, "success": 0.7},
    {"name": "Bank Heist", "energy": 60, "reward": (500, 1000), "exp": 50, "success": 0.4},
]

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="City of Syndicates API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# DEPENDENCIES / HELPERS
# =========================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_player(username: str, db: Session) -> PlayerDB:
    player = db.query(PlayerDB).filter(PlayerDB.username == username).first()
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    return player


def add_exp(player: PlayerDB, amount: int):
    player.experience += amount
    new_level = 1 + (player.experience // 100)
    if new_level > player.level:
        player.level = new_level
        player.max_energy += 10
        # little “level up” refill bonus
        player.energy = min(player.max_energy, player.energy + 20)


def player_public(player: PlayerDB) -> Dict[str, Any]:
    return {
        "username": player.username,
        "money": round(player.money, 2),
        "energy": player.energy,
        "max_energy": player.max_energy,
        "level": player.level,
        "experience": player.experience,
        "wins": player.wins,
        "losses": player.losses,
        "strength": player.strength,
        "agility": player.agility,
        "intelligence": player.intelligence,
        "charisma": player.charisma,
        "created_at": player.created_at.isoformat() if player.created_at else None,
    }


def inventory_payload(player: PlayerDB) -> Dict[str, Any]:
    items = []
    equipped = {"weapon": None, "armor": None}

    for it in player.inventory:
        items.append(
            {"item_type": it.item_type, "item_id": it.item_id, "equipped": it.equipped}
        )
        if it.equipped and it.item_type in equipped:
            equipped[it.item_type] = it.item_id

    return {"equipped": equipped, "items": items}


def ensure_starter_gear(player: PlayerDB, db: Session):
    # Give starter weapon if inventory empty
    if not player.inventory:
        starter = InventoryDB(player_id=player.id, item_type="weapon", item_id="brass_knuckles", equipped=True)
        db.add(starter)
        db.commit()


# =========================
# SCHEMAS
# =========================
class AuthIn(BaseModel):
    username: str
    password: str


class Action(BaseModel):
    username: str


class BuyItem(BaseModel):
    username: str
    item_id: str
    item_type: str  # weapon or armor


class EquipItem(BaseModel):
    username: str
    item_id: str
    item_type: str  # weapon or armor


# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def root():
    return {"status": "City of Syndicates backend online"}


@app.post("/register")
def register(data: AuthIn, db: Session = Depends(get_db)):
    existing = db.query(PlayerDB).filter(PlayerDB.username == data.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    player = PlayerDB(
        username=data.username,
        password_hash=hash_password(data.password),
        money=100.0,
        energy=100,
        max_energy=100,
    )
    db.add(player)
    db.commit()
    db.refresh(player)

    # Starter gear
    db.add(InventoryDB(player_id=player.id, item_type="weapon", item_id="brass_knuckles", equipped=True))
    db.commit()

    return {"message": "Registered", "player": player_public(player)}


@app.post("/login")
def login(data: AuthIn, db: Session = Depends(get_db)):
    user = db.query(PlayerDB).filter(PlayerDB.username == data.username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    ensure_starter_gear(user, db)

    token = make_token(user.username)
    return {"message": "Login successful", "access_token": token, "player": player_public(user)}


@app.get("/stats/{username}")
def stats(username: str, db: Session = Depends(get_db)):
    player = get_player(username, db)
    ensure_starter_gear(player, db)
    return {"player": player_public(player), "inventory": inventory_payload(player)}


@app.post("/crime")
def crime(action: Action, db: Session = Depends(get_db)):
    user = get_player(action.username, db)
    ensure_starter_gear(user, db)

    crime_pick = random.choice(CRIMES)
    if user.energy < crime_pick["energy"]:
        raise HTTPException(status_code=400, detail="Not enough energy")

    user.energy -= crime_pick["energy"]

    if random.random() <= crime_pick["success"]:
        reward = random.randint(*crime_pick["reward"])
        user.money += reward
        user.wins += 1
        add_exp(user, crime_pick["exp"])
        db.commit()
        return {
            "result": "SUCCESS",
            "crime": crime_pick["name"],
            "reward": reward,
            "player": player_public(user),
        }

    user.losses += 1
    db.commit()
    return {
        "result": "FAILED",
        "crime": crime_pick["name"],
        "player": player_public(user),
    }


@app.post("/rest")
def rest(action: Action, db: Session = Depends(get_db)):
    user = get_player(action.username, db)
    ensure_starter_gear(user, db)

    user.energy = min(user.max_energy, user.energy + 40)
    db.commit()
    return {"message": "Rested", "player": player_public(user)}


@app.get("/armory")
def armory():
    return {"weapons": WEAPONS, "armor": ARMOR}


@app.post("/armory/buy")
def buy_item(data: BuyItem, db: Session = Depends(get_db)):
    player = get_player(data.username, db)
    ensure_starter_gear(player, db)

    item_type = data.item_type.lower().strip()
    if item_type not in ("weapon", "armor"):
        raise HTTPException(status_code=400, detail="item_type must be 'weapon' or 'armor'")

    item = WEAPONS.get(data.item_id) if item_type == "weapon" else ARMOR.get(data.item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    if player.money < item["price"]:
        raise HTTPException(status_code=400, detail="Not enough money")

    # Deduct money
    player.money -= float(item["price"])

    # Unequip old of same type
    db.query(InventoryDB).filter(
        InventoryDB.player_id == player.id,
        InventoryDB.item_type == item_type
    ).update({"equipped": False})

    # Add new equipped item
    db.add(InventoryDB(
        player_id=player.id,
        item_type=item_type,
        item_id=data.item_id,
        equipped=True
    ))

    db.commit()
    db.refresh(player)

    return {
        "message": f"Bought {item['name']}",
        "player": player_public(player),
        "inventory": inventory_payload(player),
    }


@app.get("/inventory/{username}")
def get_inventory(username: str, db: Session = Depends(get_db)):
    player = get_player(username, db)
    ensure_starter_gear(player, db)
    return {"username": player.username, **inventory_payload(player)}


@app.post("/inventory/equip")
def equip_item(data: EquipItem, db: Session = Depends(get_db)):
    player = get_player(data.username, db)
    ensure_starter_gear(player, db)

    item_type = data.item_type.lower().strip()
    if item_type not in ("weapon", "armor"):
        raise HTTPException(status_code=400, detail="item_type must be 'weapon' or 'armor'")

    owned = db.query(InventoryDB).filter(
        InventoryDB.player_id == player.id,
        InventoryDB.item_type == item_type,
        InventoryDB.item_id == data.item_id
    ).first()

    if not owned:
        raise HTTPException(status_code=400, detail="Item not owned")

    db.query(InventoryDB).filter(
        InventoryDB.player_id == player.id,
        InventoryDB.item_type == item_type
    ).update({"equipped": False})

    owned.equipped = True
    db.commit()
    db.refresh(player)

    return {"message": "Equipped", "inventory": inventory_payload(player)}


@app.get("/leaderboard")
def leaderboard(limit: int = 20, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 100))
    players = db.query(PlayerDB).order_by(
        PlayerDB.level.desc(),
        PlayerDB.experience.desc(),
        PlayerDB.money.desc()
    ).limit(limit).all()

    return {"leaders": [player_public(p) for p in players]}

