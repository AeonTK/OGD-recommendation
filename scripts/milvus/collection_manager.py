import sys
import logging
from typing import List, Optional
from pymilvus import connections, utility, Collection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

def connect() -> None:
    """Connect to local Milvus."""
    try:
        connections.connect(alias="default", host="localhost", port="19530")
        logger.info("✅ Successfully connected to Milvus at localhost:19530")
        print("✅ Connected to Milvus (localhost:19530)")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Milvus: {e}")
        print(f"❌ Connection failed: {e}")
        sys.exit(1)

def list_collections_status() -> List[str]:
    """Prints a formatted table of all collections and their states."""
    try:
        cols = utility.list_collections()
        logger.debug(f"Retrieved {len(cols)} collections from Milvus")
        if not cols:
            logger.info("No collections found in the database")
            print("\n   (No collections found)")
            return []
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        print(f"❌ Failed to list collections: {e}")
        return []
    
    print(f"\n{'#':<3} | {'Name':<30} | {'State':<10} | {'Entities':<10} | {'Aliases'}")
    print("-" * 80)
    
    for i, name in enumerate(cols):
        state_code = utility.load_state(name)
        # 3=Loaded, 2=Loading, 1=NotLoad, 0=NotExist
        state_str = "🟢 LOADED" if state_code == 3 else "⚪ DISK"
        
        try:
            c = Collection(name)
            count = c.num_entities
            aliases = [a for a in utility.list_aliases(name)]
            logger.debug(f"Collection '{name}': {count} entities, state: {state_code}")
        except Exception as e:
            logger.warning(f"Could not get details for collection '{name}': {e}")
            count = "?"
            aliases = []

        alias_str = ", ".join(aliases)
        print(f"{i+1:<3} | {name:<30} | {state_str:<10} | {count:<10} | {alias_str}")
    
    print("-" * 80)
    return cols

def select_collection(cols: List[str]) -> Optional[str]:
    """Helper to let user select a collection by index."""
    if not cols: 
        logger.debug("No collections available for selection")
        return None
    try:
        idx = int(input("\nEnter number # to select collection (or 0 to cancel): "))
        if idx == 0: 
            logger.debug("User cancelled collection selection")
            return None
        if 1 <= idx <= len(cols):
            selected = cols[idx-1]
            logger.debug(f"User selected collection: {selected}")
            return selected
        else:
            logger.warning(f"User entered invalid collection number: {idx}")
            print("❌ Invalid number.")
    except ValueError as e:
        logger.warning(f"Invalid input during collection selection: {e}")
        print("❌ Invalid input.")
    return None

def load_coll(cols: List[str]) -> None:
    """Load a selected collection into memory."""
    name = select_collection(cols)
    if name:
        try:
            print(f"⏳ Loading '{name}' into RAM...")
            logger.info(f"Loading collection '{name}' into memory")
            Collection(name).load()
            logger.info(f"Successfully loaded collection '{name}'")
            print(f"✅ '{name}' is now LOADED.")
        except Exception as e:
            logger.error(f"Failed to load collection '{name}': {e}")
            print(f"❌ Failed to load '{name}': {e}")

def release_coll(cols: List[str]) -> None:
    """Release a selected collection from memory."""
    name = select_collection(cols)
    if name:
        try:
            print(f"⏳ Releasing '{name}' from RAM...")
            logger.info(f"Releasing collection '{name}' from memory")
            Collection(name).release()
            logger.info(f"Successfully released collection '{name}'")
            print(f"✅ '{name}' is now RELEASED (Memory freed).")
        except Exception as e:
            logger.error(f"Failed to release collection '{name}': {e}")
            print(f"❌ Failed to release '{name}': {e}")

def drop_coll(cols: List[str]) -> None:
    """Drop a selected collection permanently."""
    name = select_collection(cols)
    if name:
        confirm = input(f"⚠️  WARNING: This will PERMANENTLY DELETE '{name}' and all its data.\nType the collection name to confirm: ")
        if confirm == name:
            try:
                print(f"🗑️  Dropping '{name}'...")
                logger.warning(f"Dropping collection '{name}' permanently")
                utility.drop_collection(name)
                logger.info(f"Successfully dropped collection '{name}'")
                print("✅ Collection deleted.")
            except Exception as e:
                logger.error(f"Failed to drop collection '{name}': {e}")
                print(f"❌ Failed to delete '{name}': {e}")
        else:
            logger.info(f"User cancelled deletion of collection '{name}'")
            print("❌ Confirmation failed. Aborted.")

def show_schema(cols: List[str]) -> None:
    """Display the schema of a selected collection."""
    name = select_collection(cols)
    if name:
        try:
            c = Collection(name)
            logger.debug(f"Displaying schema for collection '{name}'")
            print(f"\n🔍 SCHEMA FOR: {name}")
            for field in c.schema.fields:
                print(f"   - {field.name} (Type: {field.dtype}, Params: {field.params})")
            print(f"   - Description: {c.description}")
            input("\nPress Enter to continue...")
        except Exception as e:
            logger.error(f"Failed to get schema for collection '{name}': {e}")
            print(f"❌ Failed to get schema for '{name}': {e}")

def rename_coll(cols: List[str]) -> None:
    """Rename a selected collection."""
    name = select_collection(cols)
    if name:
        new_name = input(f"Enter NEW name for '{name}': ").strip()
        if new_name:
            try:
                print(f"📝 Renaming '{name}' -> '{new_name}'...")
                logger.info(f"Renaming collection '{name}' to '{new_name}'")
                utility.rename_collection(name, new_name)
                logger.info(f"Successfully renamed collection '{name}' to '{new_name}'")
                print("✅ Renamed.")
            except Exception as e:
                logger.error(f"Failed to rename collection '{name}' to '{new_name}': {e}")
                print(f"❌ Failed to rename '{name}': {e}")
        else:
            logger.debug("User cancelled rename operation")

def manage_aliases(cols: List[str]) -> None:
    """Manage aliases for a selected collection."""
    name = select_collection(cols)
    if name:
        try:
            curr_aliases = utility.list_aliases(name)
            logger.debug(f"Current aliases for '{name}': {curr_aliases}")
            print(f"\nCurrent aliases: {curr_aliases}")
            action = input(" (A)dd alias or (D)rop alias? ").lower()
            
            if action == 'a':
                alias = input("Enter new alias name: ").strip()
                if alias:
                    try:
                        utility.create_alias(name, alias)
                        logger.info(f"Created alias '{alias}' for collection '{name}'")
                        print("✅ Alias created.")
                    except Exception as e:
                        logger.error(f"Failed to create alias '{alias}' for '{name}': {e}")
                        print(f"❌ Failed to create alias: {e}")
            elif action == 'd':
                alias = input("Enter alias to remove: ").strip()
                if alias:
                    try:
                        utility.drop_alias(alias)
                        logger.info(f"Dropped alias '{alias}'")
                        print("✅ Alias dropped.")
                    except Exception as e:
                        logger.error(f"Failed to drop alias '{alias}': {e}")
                        print(f"❌ Failed to drop alias: {e}")
        except Exception as e:
            logger.error(f"Failed to manage aliases for collection '{name}': {e}")
            print(f"❌ Failed to manage aliases: {e}")

def main_menu() -> None:
    """Main interactive menu for collection management."""
    logger.info("Starting Milvus Collection Manager")
    connect()
    
    try:
        while True:
            cols = list_collections_status()
            
            print("\nACTIONS:")
            print("1. [Load]    Fill RAM (Searchable)")
            print("2. [Release] Free RAM (Not Searchable)")
            print("3. [Drop]    Delete Collection Permanently")
            print("4. [Schema]  View Fields & Dimensions")
            print("5. [Rename]  Rename Collection")
            print("6. [Alias]   Manage Aliases")
            print("Q. Quit")
            
            choice = input("\nSelect Action: ").lower().strip()
            logger.debug(f"User selected action: {choice}")
            
            if choice == '1': 
                load_coll(cols)
            elif choice == '2': 
                release_coll(cols)
            elif choice == '3': 
                drop_coll(cols)
            elif choice == '4': 
                show_schema(cols)
            elif choice == '5': 
                rename_coll(cols)
            elif choice == '6': 
                manage_aliases(cols)
            elif choice == 'q': 
                logger.info("User quit the application")
                print("Bye!")
                break
            else:
                logger.warning(f"User entered invalid choice: {choice}")
                print("Invalid choice")
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n\n🛑 Interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"Unexpected error in main menu: {e}")
        print(f"❌ Unexpected error: {e}")

def main() -> None:
    """Main entry point for the script."""
    try:
        main_menu()
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()