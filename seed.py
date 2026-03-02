import os
from sqlalchemy.orm import Session
from main import (
    SessionLocal, engine, Base, 
    COSMETICS, WEAPONS, ARMOR, NPC_VENDORS, 
    grant_cosmetic, PlayerDB, InventoryDB
)

def seed_database():
    print("--- Starting Database Seed ---")
    db: Session = SessionLocal()

    try:
        # 1. Ensure tables exist
        Base.metadata.create_all(bind=engine)
        print("[1/3] Database tables verified.")

        # 2. Check for an Admin/System User
        admin_username = "SystemAdmin"
        admin = db.query(PlayerDB).filter(PlayerDB.username == admin_username).first()
        
        if not admin:
            print(f"[2/3] Creating system admin: {admin_username}...")
            admin = PlayerDB(
                username=admin_username,
                password_hash="SECRET_SYSTEM_HASH", # Should be hashed if used for login
                money=1000000.0,
                level=99,
                strength=100,
                agility=100,
                intelligence=100,
                charisma=100
            )
            db.add(admin)
            db.commit()
            db.refresh(admin)
            
            # Give admin a starter kit
            db.add(InventoryDB(
                player_id=admin.id, 
                item_type="weapon", 
                item_id="pistol", 
                equipped=True
            ))
            print(f"      Admin created with ID: {admin.id}")
        else:
            print("[2/3] System admin already exists.")

        # 3. Validation Summary
        print("[3/3] Validating Catalogs...")
        print(f"      - {len(COSMETICS)} Cosmetics available.")
        print(f"      - {len(WEAPONS)} Weapons available.")
        print(f"      - {len(NPC_VENDORS)} Vendors ready.")

        db.commit()
        print("--- Seed Complete: City of Syndicates is ready for players ---")

    except Exception as e:
        print(f"ERROR DURING SEED: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
