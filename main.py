import os
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    ForeignKey, Boolean
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship

from passlib.context import CryptContext
from jose import jwt, JWTError


# =========================
# CONFIG
# =========================
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"

ADMIN_KEY = os.getenv("ADMIN_KEY", "dev-admin-key-change-me")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"sslmode": "require"},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Swagger-friendly OAuth2 password flow:
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")


# =========================
# TAB DATA (Catalogs)
# =========================
CRIMES = [
    {"id": "pickpocket", "name": "Pickpocket", "energy": 10, "reward": (20, 50), "exp": 5, "success": 0.90, "heat": 2},
    {"id": "mugging", "name": "Mugging", "energy": 20, "reward": (60, 140), "exp": 12, "success": 0.70, "heat": 5},
    {"id": "warehouse_hit", "name": "Warehouse Hit", "energy": 35, "reward": (150, 320), "exp": 25, "success": 0.55, "heat": 10},
    {"id": "bank_heist", "name": "Bank Heist", "energy": 60, "reward": (500, 1000), "exp": 50, "success": 0.40, "heat": 18},
]

JOBS = [
    {"id": "courier", "name": "Courier", "energy": 10, "payout": (25, 60), "exp": 4},
    {"id": "bouncer", "name": "Bouncer", "energy": 18, "payout": (60, 140), "exp": 10},
    {"id": "bartender", "name": "Bartender", "energy": 15, "payout": (50, 110), "exp": 8},
    {"id": "vip_host", "name": "VIP Host (18+ venue)", "energy": 25, "payout": (120, 260), "exp": 18},
]

TRAINING = [
    {"id": "gym_strength", "name": "Strength Training", "stat": "strength", "energy": 20, "cost": 60, "gain": (1, 2), "exp": 8},
    {"id": "parkour_agility", "name": "Agility Drills", "stat": "agility", "energy": 20, "cost": 60, "gain": (1, 2), "exp": 8},
    {"id": "study_intel", "name": "Study Session", "stat": "intelligence", "energy": 15, "cost": 40, "gain": (1, 2), "exp": 7},
    {"id": "social_charisma", "name": "Charisma Workshop", "stat": "charisma", "energy": 15, "cost": 40, "gain": (1, 2), "exp": 7},
]

WEAPONS = {
    "brass_knuckles": {"name": "Brass Knuckles", "damage": 5, "price": 80, "level": 1},
    "switchblade": {"name": "Switchblade", "damage": 10, "price": 220, "level": 2},
    "pistol": {"name": "Pistol", "damage": 25, "price": 2000, "level": 5},
}
ARMOR = {
    "leather_jacket": {"name": "Leather Jacket", "defense": 5, "price": 120, "level": 1},
    "kevlar_vest": {"name": "Kevlar Vest", "defense": 14, "price": 1200, "level": 4},
}

# Nightlife venues (non-explicit adult venue flavor)
NIGHTLIFE = [
    {"id": "nightclub", "name": "Neon Pulse Nightclub", "features": ["tips", "reputation", "jobs: bouncer/bartender"]},
    {"id": "strip_club", "name": "Velvet Voltage (18+)", "features": ["tips", "reputation", "jobs: vip_host/security"]},
]

# =========================
# Cosmetics + Avatar system
# =========================
COSMETIC_SLOTS = [
    "hair", "skin_tone", "clothes",
    "sunglasses", "shoes", "gloves", "jewelry",
    "hats", "jacket", "pocketbook", "tattoos"
]

COSMETICS = {
    # Hair
    "hair_basic_01": {"name": "Basic Fade", "slot": "hair", "price": 0, "level": 1, "rarity": "common"},
    "hair_neon_02": {"name": "Neon Undercut", "slot": "hair", "price": 250, "level": 2, "rarity": "rare"},

    # Skin tones (free)
    "skin_tone_01": {"name": "Skin Tone 1", "slot": "skin_tone", "price": 0, "level": 1, "rarity": "common"},
    "skin_tone_02": {"name": "Skin Tone 2", "slot": "skin_tone", "price": 0, "level": 1, "rarity": "common"},
    "skin_tone_03": {"name": "Skin Tone 3", "slot": "skin_tone", "price": 0, "level": 1, "rarity": "common"},
    "skin_tone_04": {"name": "Skin Tone 4", "slot": "skin_tone", "price": 0, "level": 1, "rarity": "common"},

    # Clothes / Jackets
    "outfit_basic_01": {"name": "Basic Outfit", "slot": "clothes", "price": 0, "level": 1, "rarity": "common"},
    "outfit_street_01": {"name": "Street Fit", "slot": "clothes", "price": 120, "level": 1, "rarity": "common"},
    "outfit_syndicate_03": {"name": "Syndicate Suit", "slot": "clothes", "price": 900, "level": 4, "rarity": "epic"},
    "jacket_leather_01": {"name": "Leather Jacket", "slot": "jacket", "price": 450, "level": 2, "rarity": "rare"},

    # Accessories
    "shades_black_01": {"name": "Black Shades", "slot": "sunglasses", "price": 150, "level": 1, "rarity": "common"},
    "shoes_runner_01": {"name": "Runner Kicks", "slot": "shoes", "price": 180, "level": 1, "rarity": "common"},
    "gloves_leather_01": {"name": "Leather Gloves", "slot": "gloves", "price": 220, "level": 2, "rarity": "rare"},
    "chain_silver_01": {"name": "Silver Chain", "slot": "jewelry", "price": 300, "level": 2, "rarity": "rare"},
    "hat_cap_01": {"name": "Street Cap", "slot": "hats", "price": 110, "level": 1, "rarity": "common"},
    "pocketbook_neon_01": {"name": "Neon Pocketbook", "slot": "pocketbook", "price": 500, "level": 3, "rarity": "rare"},

    # Tattoos (non-explicit)
    "tattoo_dragon_01": {"name": "Dragon Ink", "slot": "tattoos", "price": 600, "level": 3, "rarity": "epic"},
}

NPC_VENDORS = {
    "stylist_mina": {"name": "Mina the Stylist", "min_level": 1, "sells": ["hair_basic_01", "hair_neon_02", "outfit_street_01", "hat_cap_01"]},
    "shade_dealer": {"name": "Shade Dealer", "min_level": 1, "sells": ["shades_black_01"]},
    "shoe_runner": {"name": "Runner", "min_level": 1, "sells": ["shoes_runner_01"]},
    "boutique_boss": {"name": "Boutique Boss", "min_level": 2, "sells": ["jacket_leather_01", "chain_silver_01", "pocketbook_neon_01"]},
    "ink_master": {"name": "Ink Master (18+)", "min_level": 3, "sells": ["tattoo_dragon_01"]},
}

# Quests can reward cosmetics
QUESTS = [
    {"id": "daily_crimes_3", "type": "daily", "name": "Warm-Up Hustle", "desc": "Complete 3 crimes.", "goal": {"crimes": 3}, "reward": {"money": 200, "exp": 25}},
    {"id": "daily_jobs_2", "type": "daily", "name": "Clock In", "desc": "Complete 2 jobs.", "goal": {"jobs": 2}, "reward": {"money": 160, "exp": 20}},
    {"id": "story_first_blood", "type": "story", "name": "First Blood", "desc": "Win 1 PvP match.", "goal": {"pvp_wins": 1}, "reward": {"money": 350, "exp": 40, "cosmetics": ["shades_black_01"]}},
    {"id": "story_style_1", "type": "story", "name": "Fresh Look", "desc": "Complete 2 jobs and 1 crime.", "goal": {"jobs": 2, "crimes": 1}, "reward": {"money": 150, "exp": 20, "cosmetics": ["hat_cap_01"]}},
]


# =========================
# DATABASE MODELS
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

    heat = Column(Integer, default=0)
    reputation = Column(Integer, default=0)

    # Base appearance (fallbacks if slot not equipped)
    base_hair = Column(String(50), default="hair_basic_01")
    base_skin_tone = Column(String(50), default="skin_tone_01")
    base_clothes = Column(String(50), default="outfit_basic_01")

    created_at = Column(DateTime, default=datetime.utcnow)

    inventory = relationship("InventoryDB", back_populates="player", cascade="all, delete-orphan")
    quests = relationship("PlayerQuestDB", back_populates="player", cascade="all, delete-orphan")


class InventoryDB(Base):
    __tablename__ = "inventory"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), index=True)

    item_type = Column(String(20), index=True)  # weapon/armor/cosmetic
    item_id = Column(String(100), index=True)
    slot = Column(String(30), index=True, nullable=True)  # cosmetic slot e.g. hats, shoes
    equipped = Column(Boolean, default=False)
    acquired_at = Column(DateTime, default=datetime.utcnow)

    player = relationship("PlayerDB", back_populates="inventory")


class PlayerQuestDB(Base):
    __tablename__ = "player_quests"

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"), index=True)
    quest_id = Column(String(100), index=True)
    quest_type = Column(String(20))
    status = Column(String(20), default="active")  # active/completed/claimed

    crimes_done = Column(Integer, default=0)
    jobs_done = Column(Integer, default=0)
    pvp_wins = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    player = relationship("PlayerDB", back_populates="quests")


class PvPMatchDB(Base):
    __tablename__ = "pvp_matches"

    id = Column(Integer, primary_key=True)
    status = Column(String(20), default="open")  # open/active/finished

    player_a = Column(String(50), index=True)
    player_b = Column(String(50), index=True, nullable=True)

    a_hp = Column(Integer, default=100)
    b_hp = Column(Integer, default=100)

    turn = Column(String(1), default="A")  # A/B
    winner = Column(String(50), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)


class CasinoLedgerDB(Base):
    __tablename__ = "casino_ledger"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), index=True)
    game = Column(String(50))
    wager = Column(Integer)
    delta = Column(Integer)  # +win or -loss
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="City of Syndicates API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# HELPERS
# =========================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(pw: str) -> str:
    return pwd_context.hash(pw)


def verify_password(pw: str, hashed: str) -> bool:
    return pwd_context.verify(pw, hashed)


def add_exp(p: PlayerDB, amount: int):
    p.experience += amount
    new_level = 1 + (p.experience // 100)
    if new_level > p.level:
        p.level = new_level
        p.max_energy += 10
        p.energy = min(p.max_energy, p.energy + 20)


def player_public(p: PlayerDB) -> Dict[str, Any]:
    return {
        "username": p.username,
        "money": round(p.money, 2),
        "energy": p.energy,
        "max_energy": p.max_energy,
        "level": p.level,
        "experience": p.experience,
        "wins": p.wins,
        "losses": p.losses,
        "heat": p.heat,
        "reputation": p.reputation,
        "strength": p.strength,
        "agility": p.agility,
        "intelligence": p.intelligence,
        "charisma": p.charisma,
    }


def inventory_payload(p: PlayerDB) -> Dict[str, Any]:
    equipped_core = {"weapon": None, "armor": None}
    cosmetics_equipped = {slot: None for slot in COSMETIC_SLOTS}
    items = []

    for it in p.inventory:
        items.append({"item_type": it.item_type, "item_id": it.item_id, "slot": it.slot, "equipped": it.equipped})

        if it.equipped and it.item_type in equipped_core:
            equipped_core[it.item_type] = it.item_id

        if it.equipped and it.item_type == "cosmetic" and it.slot in cosmetics_equipped:
            cosmetics_equipped[it.slot] = it.item_id

    return {"equipped_core": equipped_core, "cosmetics_equipped": cosmetics_equipped, "items": items}


def get_player(username: str, db: Session) -> PlayerDB:
    p = db.query(PlayerDB).filter(PlayerDB.username == username).first()
    if not p:
        raise HTTPException(status_code=404, detail="Player not found")
    return p


def ensure_starter(p: PlayerDB, db: Session):
    # Starter weapon if empty
    owned_weapon = db.query(InventoryDB).filter(
        InventoryDB.player_id == p.id,
        InventoryDB.item_type == "weapon"
    ).first()
    if not owned_weapon:
        db.add(InventoryDB(player_id=p.id, item_type="weapon", item_id="brass_knuckles", slot=None, equipped=True))
        db.commit()


def make_token(username: str) -> str:
    payload = {"sub": username, "exp": datetime.utcnow() + timedelta(days=7)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> PlayerDB:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    p = get_player(username, db)
    ensure_starter(p, db)
    ensure_player_quests(p, db)
    return p


def require_admin(x_admin_key: Optional[str] = Header(default=None)):
    if not x_admin_key or x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Admin key required")


def get_item(item_type: str, item_id: str) -> Optional[Dict[str, Any]]:
    t = item_type.lower().strip()
    if t == "weapon":
        return WEAPONS.get(item_id)
    if t == "armor":
        return ARMOR.get(item_id)
    return None


def get_cosmetic(cosmetic_id: str) -> Optional[Dict[str, Any]]:
    return COSMETICS.get(cosmetic_id)


def ensure_player_quests(p: PlayerDB, db: Session):
    existing = {q.quest_id for q in p.quests}
    for q in QUESTS:
        if q["id"] not in existing:
            db.add(PlayerQuestDB(player_id=p.id, quest_id=q["id"], quest_type=q["type"], status="active"))
    db.commit()
    db.refresh(p)


def quest_payload(p: PlayerDB) -> List[Dict[str, Any]]:
    by_id = {q["id"]: q for q in QUESTS}
    out = []
    for pq in p.quests:
        qdef = by_id.get(pq.quest_id)
        if not qdef:
            continue
        out.append({
            "quest_id": pq.quest_id,
            "type": pq.quest_type,
            "name": qdef["name"],
            "desc": qdef["desc"],
            "status": pq.status,
            "progress": {"crimes": pq.crimes_done, "jobs": pq.jobs_done, "pvp_wins": pq.pvp_wins},
            "goal": qdef["goal"],
            "reward": qdef["reward"],
        })
    return out


def sync_quest_progress(p: PlayerDB, db: Session, inc: Dict[str, int]):
    ensure_player_quests(p, db)

    for pq in p.quests:
        if pq.status != "active":
            continue
        pq.crimes_done += inc.get("crimes", 0)
        pq.jobs_done += inc.get("jobs", 0)
        pq.pvp_wins += inc.get("pvp_wins", 0)

        qdef = next((q for q in QUESTS if q["id"] == pq.quest_id), None)
        if not qdef:
            continue
        goal = qdef["goal"]
        done = (
            pq.crimes_done >= goal.get("crimes", 0)
            and pq.jobs_done >= goal.get("jobs", 0)
            and pq.pvp_wins >= goal.get("pvp_wins", 0)
        )
        if done:
            pq.status = "completed"
            pq.completed_at = datetime.utcnow()

    db.commit()


def grant_cosmetic(db: Session, player: PlayerDB, cosmetic_id: str, auto_equip: bool = False) -> bool:
    c = get_cosmetic(cosmetic_id)
    if not c:
        return False

    owned = db.query(InventoryDB).filter(
        InventoryDB.player_id == player.id,
        InventoryDB.item_type == "cosmetic",
        InventoryDB.item_id == cosmetic_id
    ).first()
    if owned:
        return False

    if auto_equip:
        db.query(InventoryDB).filter(
            InventoryDB.player_id == player.id,
            InventoryDB.item_type == "cosmetic",
            InventoryDB.slot == c["slot"]
        ).update({"equipped": False})

    db.add(InventoryDB(
        player_id=player.id,
        item_type="cosmetic",
        item_id=cosmetic_id,
        slot=c["slot"],
        equipped=bool(auto_equip)
    ))
    return True


def avatar_snapshot(player: PlayerDB) -> Dict[str, Any]:
    equipped = {slot: None for slot in COSMETIC_SLOTS}
    for it in player.inventory:
        if it.item_type == "cosmetic" and it.equipped and it.slot in equipped:
            equipped[it.slot] = it.item_id

    base = {
        "hair": equipped["hair"] or player.base_hair,
        "skin_tone": equipped["skin_tone"] or player.base_skin_tone,
        "clothes": equipped["clothes"] or player.base_clothes,
    }

    equipped_out = equipped.copy()
    equipped_out.pop("hair", None)
    equipped_out.pop("skin_tone", None)
    equipped_out.pop("clothes", None)

    return {"base": base, "equipped": equipped_out}


def pvp_power(p: PlayerDB) -> int:
    return (p.strength * 3) + (p.agility * 2) + p.intelligence + p.charisma


# =========================
# SCHEMAS
# =========================
class AuthIn(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=4, max_length=128)


class BuyItemIn(BaseModel):
    item_type: str
    item_id: str


class EquipItemIn(BaseModel):
    item_type: str
    item_id: str


class PvPActionIn(BaseModel):
    match_id: int
    action: str = Field(default="attack")  # attack/defend


class SlotsIn(BaseModel):
    wager: int = Field(ge=1, le=10000)


class BlackjackIn(BaseModel):
    wager: int = Field(ge=1, le=10000)


class NightlifeTipIn(BaseModel):
    venue_id: str
    amount: int = Field(ge=1, le=5000)


class CosmeticBuyIn(BaseModel):
    cosmetic_id: str


class CosmeticEquipIn(BaseModel):
    cosmetic_id: str


class AppearanceSetIn(BaseModel):
    base_hair: Optional[str] = None
    base_skin_tone: Optional[str] = None
    base_clothes: Optional[str] = None


class AdminMoneyIn(BaseModel):
    username: str
    amount: int


class AdminGiveItemIn(BaseModel):
    username: str
    item_type: str
    item_id: str
    equip: bool = True


class AdminResetIn(BaseModel):
    username: str


# =========================
# PUBLIC / AUTH
# =========================
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/register")
def register(data: AuthIn, db: Session = Depends(get_db)):
    if db.query(PlayerDB).filter(PlayerDB.username == data.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")

    p = PlayerDB(
        username=data.username,
        password_hash=hash_password(data.password),
        money=100.0,
        energy=100,
        max_energy=100,
    )
    db.add(p)
    db.commit()
    db.refresh(p)

    # Starter weapon
    db.add(InventoryDB(player_id=p.id, item_type="weapon", item_id="brass_knuckles", slot=None, equipped=True))

    # Starter cosmetics as owned (optional)
    grant_cosmetic(db, p, "hair_basic_01", auto_equip=True)
    grant_cosmetic(db, p, "skin_tone_01", auto_equip=True)
    grant_cosmetic(db, p, "outfit_basic_01", auto_equip=True)

    db.commit()
    ensure_player_quests(p, db)
    return {"message": "Registered", "player": player_public(p)}


@app.post("/login")
def login(data: AuthIn, db: Session = Depends(get_db)):
    p = db.query(PlayerDB).filter(PlayerDB.username == data.username).first()
    if not p or not verify_password(data.password, p.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    ensure_starter(p, db)
    ensure_player_quests(p, db)
    token = make_token(p.username)
    return {"message": "Login successful", "access_token": token, "player": player_public(p)}


# Swagger auth: type username/password, Swagger stores bearer token automatically
@app.post("/token")
def token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    p = db.query(PlayerDB).filter(PlayerDB.username == form_data.username).first()
    if not p or not verify_password(form_data.password, p.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    ensure_starter(p, db)
    ensure_player_quests(p, db)
    access_token = make_token(p.username)
    return {"access_token": access_token, "token_type": "bearer"}


# =========================
# HUD: STATS / INVENTORY / AVATAR
# =========================
@app.get("/stats")
def stats(me: PlayerDB = Depends(get_current_user)):
    return {
        "player": player_public(me),
        "inventory": inventory_payload(me),
        "quests": quest_payload(me),
        "avatar": avatar_snapshot(me),
    }


@app.get("/inventory")
def inventory(me: PlayerDB = Depends(get_current_user)):
    return {"inventory": inventory_payload(me)}


@app.get("/avatar")
def get_avatar(me: PlayerDB = Depends(get_current_user)):
    return {"avatar": avatar_snapshot(me), "player": player_public(me)}


@app.post("/avatar/base")
def set_base_appearance(data: AppearanceSetIn, me: PlayerDB = Depends(get_current_user), db: Session = Depends(get_db)):
    if data.base_hair:
        c = get_cosmetic(data.base_hair)
        if not c or c["slot"] != "hair":
            raise HTTPException(status_code=400, detail="Invalid base_hair")
        me.base_hair = data.base_hair

    if data.base_skin_tone:
        c = get_cosmetic(data.base_skin_tone)
        if not c or c["slot"] != "skin_tone":
            raise HTTPException(status_code=400, detail="Invalid base_skin_tone")
        me.base_skin_tone = data.base_skin_tone

    if data.base_clothes:
        c = get_cosmetic(data.base_clothes)
        if not c or c["slot"] != "clothes":
            raise HTTPException(status_code=400, detail="Invalid base_clothes")
        me.base_clothes = data.base_clothes

    db.commit()
    db.refresh(me)
    return {"message": "Base appearance updated", "avatar": avatar_snapshot(me)}


# =========================
# TAB: CRIMES
# =========================
@app.get("/crimes")
def crimes_catalog():
    return {"crimes": CRIMES}


@app.post("/crimes/do/{crime_id}")
def crimes_do(crime_id: str, me: PlayerDB = Depends(get_current_user), db: Session = Depends(get_db)):
    c = next((x for x in CRIMES if x["id"] == crime_id), None)
    if not c:
        raise HTTPException(status_code=404, detail="Crime not found")
    if me.energy < c["energy"]:
        raise HTTPException(status_code=400, detail="Not enough energy")

    me.energy -= c["energy"]
    me.heat += c["heat"]

    if random.random() <= c["success"]:
        reward = random.randint(*c["reward"])
        me.money += reward
        me.wins += 1
        add_exp(me, c["exp"])
        sync_quest_progress(me, db, {"crimes": 1})
        db.commit()
        return {"result": "SUCCESS", "crime": c["name"], "reward": reward, "player": player_public(me)}

    me.losses += 1
    sync_quest_progress(me, db, {"crimes": 1})
    db.commit()
    return {"result": "FAILED", "crime": c["name"], "player": player_public(me)}


# =========================
# TAB: JOBS
# =========================
@app.get("/jobs")
def jobs_catalog():
    return {"jobs": JOBS}


@app.post("/jobs/do/{job_id}")
def jobs_do(job_id: str, me: PlayerDB = Depends(get_current_user), db: Session = Depends(get_db)):
    j = next((x for x in JOBS if x["id"] == job_id), None)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    if me.energy < j["energy"]:
        raise HTTPException(status_code=400, detail="Not enough energy")

    me.energy -= j["energy"]
    payout = random.randint(*j["payout"])
    me.money += payout
    add_exp(me, j["exp"])

    rep_gain = 1 if job_id in ("bartender", "vip_host") else 0
    me.reputation += rep_gain

    sync_quest_progress(me, db, {"jobs": 1})
    db.commit()
    return {"result": "PAID", "job": j["name"], "payout": payout, "rep_gain": rep_gain, "player": player_public(me)}


# =========================
# TAB: TRAINING
# =========================
@app.get("/training")
def training_catalog():
    return {"training": TRAINING}


@app.post("/training/do/{training_id}")
def training_do(training_id: str, me: PlayerDB = Depends(get_current_user), db: Session = Depends(get_db)):
    t = next((x for x in TRAINING if x["id"] == training_id), None)
    if not t:
        raise HTTPException(status_code=404, detail="Training not found")
    if me.energy < t["energy"]:
        raise HTTPException(status_code=400, detail="Not enough energy")
    if me.money < t["cost"]:
        raise HTTPException(status_code=400, detail="Not enough money")

    me.energy -= t["energy"]
    me.money -= t["cost"]
    gain = random.randint(*t["gain"])

    stat = t["stat"]
    if stat == "strength":
        me.strength += gain
    elif stat == "agility":
        me.agility += gain
    elif stat == "intelligence":
        me.intelligence += gain
    elif stat == "charisma":
        me.charisma += gain
    else:
        raise HTTPException(status_code=500, detail="Invalid training config")

    add_exp(me, t["exp"])
    db.commit()
    return {"result": "TRAINED", "training": t["name"], "stat": stat, "gain": gain, "player": player_public(me)}


# =========================
# TAB: ARMORY
# =========================
@app.get("/armory")
def armory():
    return {"weapons": WEAPONS, "armor": ARMOR}


@app.post("/armory/buy")
def armory_buy(data: BuyItemIn, me: PlayerDB = Depends(get_current_user), db: Session = Depends(get_db)):
    item_type = data.item_type.lower().strip()
    if item_type not in ("weapon", "armor"):
        raise HTTPException(status_code=400, detail="item_type must be weapon or armor")

    item = get_item(item_type, data.item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    if me.level < int(item.get("level", 1)):
        raise HTTPException(status_code=400, detail="Level too low")

    price = float(item["price"])
    if me.money < price:
        raise HTTPException(status_code=400, detail="Not enough money")

    me.money -= price

    db.query(InventoryDB).filter(
        InventoryDB.player_id == me.id,
        InventoryDB.item_type == item_type
    ).update({"equipped": False})

    db.add(InventoryDB(player_id=me.id, item_type=item_type, item_id=data.item_id, slot=None, equipped=True))
    db.commit()
    db.refresh(me)

    return {"message": f"Bought {item['name']}", "player": player_public(me), "inventory": inventory_payload(me)}


@app.post("/inventory/equip")
def equip_core(data: EquipItemIn, me: PlayerDB = Depends(get_current_user), db: Session = Depends(get_db)):
    item_type = data.item_type.lower().strip()
    if item_type not in ("weapon", "armor"):
        raise HTTPException(status_code=400, detail="item_type must be weapon or armor")

    owned = db.query(InventoryDB).filter(
        InventoryDB.player_id == me.id,
        InventoryDB.item_type == item_type,
        InventoryDB.item_id == data.item_id
    ).first()
    if not owned:
        raise HTTPException(status_code=400, detail="Item not owned")

    db.query(InventoryDB).filter(
        InventoryDB.player_id == me.id,
        InventoryDB.item_type == item_type
    ).update({"equipped": False})

    owned.equipped = True
    db.commit()
    return {"message": "Equipped", "inventory": inventory_payload(me)}


# =========================
# TAB: COSMETICS + NPC VENDORS
# =========================
@app.get("/cosmetics")
def cosmetics_catalog(slot: Optional[str] = None, me: PlayerDB = Depends(get_current_user)):
    items = []
    for cid, c in COSMETICS.items():
        if slot and c["slot"] != slot:
            continue
        if me.level < int(c.get("level", 1)):
            continue
        items.append({"id": cid, **c})
    return {"cosmetics": items, "slots": COSMETIC_SLOTS}


@app.get("/npcs/vendors")
def list_vendors():
    return {
        "vendors": [
            {"id": vid, "name": v["name"], "min_level": v["min_level"], "count": len(v["sells"])}
            for vid, v in NPC_VENDORS.items()
        ]
    }


@app.get("/npcs/vendors/{vendor_id}")
def vendor_details(vendor_id: str, me: PlayerDB = Depends(get_current_user)):
    v = NPC_VENDORS.get(vendor_id)
    if not v:
        raise HTTPException(status_code=404, detail="Vendor not found")
    if me.level < v["min_level"]:
        raise HTTPException(status_code=400, detail="Level too low for this vendor")

    inventory = []
    for cid in v["sells"]:
        c = COSMETICS.get(cid)
        if not c:
            continue
        inventory.append({"id": cid, **c})

    return {"vendor": {"id": vendor_id, "name": v["name"], "min_level": v["min_level"]}, "sells": inventory}


@app.post("/npcs/vendors/{vendor_id}/buy")
def buy_from_vendor(vendor_id: str, data: CosmeticBuyIn, me: PlayerDB = Depends(get_current_user), db: Session = Depends(get_db)):
    v = NPC_VENDORS.get(vendor_id)
    if not v:
        raise HTTPException(status_code=404, detail="Vendor not found")
    if me.level < v["min_level"]:
        raise HTTPException(status_code=400, detail="Level too low for this vendor")
    if data.cosmetic_id not in v["sells"]:
        raise HTTPException(status_code=400, detail="This vendor does not sell that item")

    c = get_cosmetic(data.cosmetic_id)
    if not c:
        raise HTTPException(status_code=404, detail="Cosmetic not found")
    if me.level < int(c.get("level", 1)):
        raise HTTPException(status_code=400, detail="Level too low for this item")

    price = int(c.get("price", 0))
    if me.money < price:
        raise HTTPException(status_code=400, detail="Not enough money")

    owned = db.query(InventoryDB).filter(
        InventoryDB.player_id == me.id,
        InventoryDB.item_type == "cosmetic",
        InventoryDB.item_id == data.cosmetic_id
    ).first()
    if owned:
        return {"message": "Already owned", "player": player_public(me), "avatar": avatar_snapshot(me)}

    me.money -= price
    grant_cosmetic(db, me, data.cosmetic_id, auto_equip=False)
    db.commit()
    db.refresh(me)

    return {"message": f"Bought {c['name']}", "player": player_public(me)}


@app.post("/cosmetics/equip")
def equip_cosmetic(data: CosmeticEquipIn, me: PlayerDB = Depends(get_current_user), db: Session = Depends(get_db)):
    c = get_cosmetic(data.cosmetic_id)
    if not c:
        raise HTTPException(status_code=404, detail="Cosmetic not found")

    owned = db.query(InventoryDB).filter(
        InventoryDB.player_id == me.id,
        InventoryDB.item_type == "cosmetic",
        InventoryDB.item_id == data.cosmetic_id
    ).first()
    if not owned:
        raise HTTPException(status_code=400, detail="Cosmetic not owned")

    slot = c["slot"]

    db.query(InventoryDB).filter(
        InventoryDB.player_id == me.id,
        InventoryDB.item_type == "cosmetic",
        InventoryDB.slot == slot
    ).update({"equipped": False})

    owned.slot = slot
    owned.equipped = True
    db.commit()
    db.refresh(me)

    return {"message": "Equipped", "avatar": avatar_snapshot(me)}


# =========================
# TAB: QUESTS
# =========================
@app.get("/quests")
def quests(me: PlayerDB = Depends(get_current_user)):
    return {"quests": quest_payload(me)}


@app.post("/quests/claim/{quest_id}")
def quests_claim(quest_id: str, me: PlayerDB = Depends(get_current_user), db: Session = Depends(get_db)):
    pq = db.query(PlayerQuestDB).filter(
        PlayerQuestDB.player_id == me.id,
        PlayerQuestDB.quest_id == quest_id
    ).first()
    if not pq:
        raise HTTPException(status_code=404, detail="Quest not found")
    if pq.status != "completed":
        raise HTTPException(status_code=400, detail="Quest not completed")

    pq.status = "claimed"
    qdef = next((q for q in QUESTS if q["id"] == quest_id), None)
    if not qdef:
        raise HTTPException(status_code=500, detail="Quest config missing")

    reward = qdef["reward"]
    me.money += reward.get("money", 0)
    add_exp(me, reward.get("exp", 0))

    granted = []
    cosmetics = reward.get("cosmetics", [])
    for cid in cosmetics:
        c = COSMETICS.get(cid)
        if c and me.level >= int(c.get("level", 1)):
            if grant_cosmetic(db, me, cid, auto_equip=False):
                granted.append(cid)

    db.commit()
    db.refresh(me)
    return {"message": "Claimed", "reward": reward, "granted_cosmetics": granted, "player": player_public(me), "avatar": avatar_snapshot(me)}


# =========================
# TAB: PVP
# =========================
@app.post("/pvp/queue")
def pvp_queue(me: PlayerDB = Depends(get_current_user), db: Session = Depends(get_db)):
    open_match = db.query(PvPMatchDB).filter(PvPMatchDB.status == "open").first()
    if open_match and open_match.player_a != me.username:
        open_match.player_b = me.username
        open_match.status = "active"
        open_match.turn = "A"
        open_match.a_hp = 100
        open_match.b_hp = 100
        db.commit()
        return {"message": "Matched!", "match_id": open_match.id, "you_are": "B"}

    m = PvPMatchDB(status="open", player_a=me.username, player_b=None, a_hp=100, b_hp=100, turn="A")
    db.add(m)
    db.commit()
    db.refresh(m)
    return {"message": "Queued. Waiting for opponent.", "match_id": m.id, "you_are": "A"}


@app.get("/pvp/match/{match_id}")
def pvp_match(match_id: int, me: PlayerDB = Depends(get_current_user), db: Session = Depends(get_db)):
    m = db.query(PvPMatchDB).filter(PvPMatchDB.id == match_id).first()
    if not m:
        raise HTTPException(status_code=404, detail="Match not found")
    if me.username not in (m.player_a, m.player_b):
        raise HTTPException(status_code=403, detail="Not your match")
    return {"match_id": m.id, "status": m.status, "player_a": m.player_a, "player_b": m.player_b, "a_hp": m.a_hp, "b_hp": m.b_hp, "turn": m.turn, "winner": m.winner}


@app.post("/pvp/act")
def pvp_act(data: PvPActionIn, me: PlayerDB = Depends(get_current_user), db: Session = Depends(get_db)):
    m = db.query(PvPMatchDB).filter(PvPMatchDB.id == data.match_id).first()
    if not m:
        raise HTTPException(status_code=404, detail="Match not found")
    if m.status != "active":
        raise HTTPException(status_code=400, detail="Match not active")
    if not m.player_b:
        raise HTTPException(status_code=400, detail="Waiting for opponent")
    if me.username not in (m.player_a, m.player_b):
        raise HTTPException(status_code=403, detail="Not your match")

    is_a = me.username == m.player_a
    if (m.turn == "A" and not is_a) or (m.turn == "B" and is_a):
        raise HTTPException(status_code=400, detail="Not your turn")

    opp_name = m.player_b if is_a else m.player_a
    opp = get_player(opp_name, db)

    action = data.action.lower().strip()
    if action not in ("attack", "defend"):
        action = "attack"

    atk = pvp_power(me)
    dfn = pvp_power(opp)
    dmg = max(5, int((atk - dfn * 0.35) / 5) + random.randint(3, 10))
    if action == "defend":
        dmg = max(1, dmg // 2)

    if is_a:
        m.b_hp = max(0, m.b_hp - dmg)
        m.turn = "B"
    else:
        m.a_hp = max(0, m.a_hp - dmg)
        m.turn = "A"

    if m.a_hp == 0 or m.b_hp == 0:
        m.status = "finished"
        m.finished_at = datetime.utcnow()
        winner = m.player_a if m.b_hp == 0 else m.player_b
        m.winner = winner

        if me.username == winner:
            me.wins += 1
            me.money += 180
            add_exp(me, 20)
            sync_quest_progress(me, db, {"pvp_wins": 1})
        else:
            me.losses += 1
            add_exp(me, 8)

    db.commit()
    return {"message": "Action processed", "damage": dmg, "match": {"match_id": m.id, "status": m.status, "a_hp": m.a_hp, "b_hp": m.b_hp, "turn": m.turn, "winner": m.winner}, "player": player_public(me)}


# =========================
# TAB: CASINO
# =========================
@app.get("/casino")
def casino_catalog():
    return {"games": ["slots", "blackjack_lite"], "limits": {"min": 1, "max": 10000}}


@app.post("/casino/slots")
def casino_slots(data: SlotsIn, me: PlayerDB = Depends(get_current_user), db: Session = Depends(get_db)):
    wager = int(data.wager)
    if me.money < wager:
        raise HTTPException(status_code=400, detail="Not enough money")
    me.money -= wager

    reel = ["🍒", "🍋", "🔔", "💎", "7️⃣"]
    spin = [random.choice(reel) for _ in range(3)]

    mult = 0
    if spin[0] == spin[1] == spin[2]:
        mult = 8 if spin[0] == "7️⃣" else 5 if spin[0] == "💎" else 3
    elif len(set(spin)) == 2:
        mult = 1

    win = int(wager * mult)
    me.money += win
    delta = win - wager

    db.add(CasinoLedgerDB(username=me.username, game="slots", wager=wager, delta=delta))
    db.commit()
    return {"spin": spin, "wager": wager, "win": win, "delta": delta, "player": player_public(me)}


@app.post("/casino/blackjack")
def casino_blackjack(data: BlackjackIn, me: PlayerDB = Depends(get_current_user), db: Session = Depends(get_db)):
    wager = int(data.wager)
    if me.money < wager:
        raise HTTPException(status_code=400, detail="Not enough money")
    me.money -= wager

    player_total = random.randint(15, 23)
    dealer_total = random.randint(16, 23)

    result = "push"
    delta = 0

    if player_total > 21 and dealer_total > 21:
        result = "push"
        me.money += wager
    elif player_total > 21:
        result = "lose"
        delta = -wager
    elif dealer_total > 21:
        result = "win"
        me.money += int(wager * 2)
        delta = int(wager)
    elif player_total > dealer_total:
        result = "win"
        me.money += int(wager * 2)
        delta = int(wager)
    elif player_total < dealer_total:
        result = "lose"
        delta = -wager
    else:
        result = "push"
        me.money += wager

    db.add(CasinoLedgerDB(username=me.username, game="blackjack_lite", wager=wager, delta=delta))
    db.commit()
    return {"player_total": player_total, "dealer_total": dealer_total, "result": result, "wager": wager, "delta": delta, "player": player_public(me)}


# =========================
# TAB: NIGHTLIFE
# =========================
@app.get("/nightlife")
def nightlife_catalog():
    return {"venues": NIGHTLIFE, "note": "Non-explicit adult venue flavor. Gate 18+ in client UI."}


@app.post("/nightlife/tip")
def nightlife_tip(data: NightlifeTipIn, me: PlayerDB = Depends(get_current_user), db: Session = Depends(get_db)):
    venue = next((v for v in NIGHTLIFE if v["id"] == data.venue_id), None)
    if not venue:
        raise HTTPException(status_code=404, detail="Venue not found")

    amt = int(data.amount)
    if me.money < amt:
        raise HTTPException(status_code=400, detail="Not enough money")

    me.money -= amt
    rep_gain = max(1, amt // 50)
    me.reputation += rep_gain
    db.commit()
    return {"message": f"Tipped at {venue['name']}", "rep_gain": rep_gain, "player": player_public(me)}


# =========================
# REST
# =========================
@app.post("/rest")
def rest(me: PlayerDB = Depends(get_current_user), db: Session = Depends(get_db)):
    me.energy = min(me.max_energy, me.energy + 40)
    me.heat = max(0, me.heat - 3)
    db.commit()
    return {"message": "Rested", "player": player_public(me)}


# =========================
# LEADERBOARD
# =========================
@app.get("/leaderboard")
def leaderboard(limit: int = 20, db: Session = Depends(get_db)):
    limit = max(1, min(limit, 100))
    players = db.query(PlayerDB).order_by(PlayerDB.level.desc(), PlayerDB.experience.desc(), PlayerDB.money.desc()).limit(limit).all()
    return {"leaders": [player_public(p) for p in players]}


# =========================
# ADMIN / CREATOR
# =========================
@app.get("/admin/players", dependencies=[Depends(require_admin)])
def admin_players(db: Session = Depends(get_db), limit: int = 50):
    limit = max(1, min(limit, 200))
    players = db.query(PlayerDB).order_by(PlayerDB.created_at.desc()).limit(limit).all()
    return {"players": [player_public(p) for p in players]}


@app.post("/admin/money", dependencies=[Depends(require_admin)])
def admin_money(data: AdminMoneyIn, db: Session = Depends(get_db)):
    p = get_player(data.username, db)
    p.money += int(data.amount)
    db.commit()
    return {"message": "Money updated", "player": player_public(p)}


@app.post("/admin/give-item", dependencies=[Depends(require_admin)])
def admin_give_item(data: AdminGiveItemIn, db: Session = Depends(get_db)):
    p = get_player(data.username, db)
    item_type = data.item_type.lower().strip()

    if item_type in ("weapon", "armor"):
        item = get_item(item_type, data.item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")

        if data.equip:
            db.query(InventoryDB).filter(InventoryDB.player_id == p.id, InventoryDB.item_type == item_type).update({"equipped": False})

        db.add(InventoryDB(player_id=p.id, item_type=item_type, item_id=data.item_id, slot=None, equipped=bool(data.equip)))
        db.commit()
        db.refresh(p)
        return {"message": "Granted", "inventory": inventory_payload(p), "avatar": avatar_snapshot(p)}

    if item_type == "cosmetic":
        c = get_cosmetic(data.item_id)
        if not c:
            raise HTTPException(status_code=404, detail="Cosmetic not found")
        ok = grant_cosmetic(db, p, data.item_id, auto_equip=bool(data.equip))
        db.commit()
        db.refresh(p)
        return {"message": "Granted" if ok else "Already owned", "inventory": inventory_payload(p), "avatar": avatar_snapshot(p)}

    raise HTTPException(status_code=400, detail="item_type must be weapon, armor, or cosmetic")


@app.post("/admin/reset-player", dependencies=[Depends(require_admin)])
def admin_reset_player(data: AdminResetIn, db: Session = Depends(get_db)):
    p = get_player(data.username, db)

    p.money = 100.0
    p.energy = 100
    p.max_energy = 100
    p.level = 1
    p.experience = 0
    p.strength = 1
    p.agility = 1
    p.intelligence = 1
    p.charisma = 1
    p.wins = 0
    p.losses = 0
    p.heat = 0
    p.reputation = 0

    p.base_hair = "hair_basic_01"
    p.base_skin_tone = "skin_tone_01"
    p.base_clothes = "outfit_basic_01"

    db.query(InventoryDB).filter(InventoryDB.player_id == p.id).delete()
    db.add(InventoryDB(player_id=p.id, item_type="weapon", item_id="brass_knuckles", slot=None, equipped=True))
    grant_cosmetic(db, p, "hair_basic_01", auto_equip=True)
    grant_cosmetic(db, p, "skin_tone_01", auto_equip=True)
    grant_cosmetic(db, p, "outfit_basic_01", auto_equip=True)

    db.query(PlayerQuestDB).filter(PlayerQuestDB.player_id == p.id).delete()
    db.commit()
    ensure_player_quests(p, db)

    return {"message": "Player reset", "player": player_public(p), "avatar": avatar_snapshot(p)}
