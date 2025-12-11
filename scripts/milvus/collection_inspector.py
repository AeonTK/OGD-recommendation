import sys
import logging
from typing import List, Tuple, Optional
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


def _print_header() -> None:
    """Print a formatted header for the inspection report."""
    header = "═" * 70
    title = "🔍 INSPECTING MILVUS COLLECTIONS"
    print(f"\n{header}")
    print(f"{title:^70}")
    print(f"{header}\n")


def _connect_to_milvus() -> bool:
    """Connect to Milvus instance."""
    try:
        connections.connect(alias="default", host="localhost", port="19530")
        logger.info("✅ Successfully connected to Milvus at localhost:19530")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to connect to Milvus: {e}")
        return False


def _get_all_collections() -> Optional[List[str]]:
    """Retrieve all collections from Milvus."""
    try:
        collections = utility.list_collections()
        logger.debug(f"Retrieved {len(collections)} collections from Milvus")
        return collections
    except Exception as e:
        logger.error(f"❌ Failed to list collections: {e}")
        return None


def _print_status_report(
    loaded_collections: List[Tuple[str, int]], 
    unloaded_collections: List[str], 
    loading_collections: List[str]
) -> None:
    """Print a formatted status report of all collections."""
    
    # Loaded collections section
    print("\n🟢 LOADED COLLECTIONS (Active in RAM)")
    print("═" * 65)
    if loaded_collections:
        print(f"{'Collection Name':<35} │ {'Entity Count':<15} │ {'Status':<10}")
        print("─" * 65)
        for name, count in loaded_collections:
            status = "✅ Ready"
            print(f"{name:<35} │ {count:<15,} │ {status:<10}")
    else:
        print("   (No loaded collections found)")
    
    # Unloaded collections section
    print("\n⚪ UNLOADED COLLECTIONS (Stored on Disk)")
    print("═" * 65)
    if unloaded_collections:
        for i, name in enumerate(unloaded_collections, 1):
            print(f"   {i:2d}. {name}")
    else:
        print("   (No unloaded collections found)")
    
    # Loading collections section (if any)
    if loading_collections:
        print("\n🟡 LOADING IN PROGRESS")
        print("═" * 65)
        for i, name in enumerate(loading_collections, 1):
            print(f"   {i:2d}. {name} (loading...)")
    
    print("\n" + "═" * 70 + "\n")


def inspect_milvus() -> None:
    """
    Inspect Milvus collections and display their status in a formatted report.
    
    This function connects to a local Milvus instance, retrieves all collections,
    checks their load status, and presents a comprehensive overview.
    """
    _print_header()

    # 1. Connect to Milvus
    if not _connect_to_milvus():
        return

    # 2. Get all collections
    all_collections = _get_all_collections()
    if all_collections is None:
        return

    if not all_collections:
        logger.info("No collections found in the database.")
        return

    loaded_collections = []
    unloaded_collections = []
    loading_collections = []

    logger.info(f"Found {len(all_collections)} total collections.")

    # 3. Check status of each collection
    logger.info("Analyzing collection status...")
    for name in all_collections:
        try:
            # utility.load_state returns an Enum or int depending on version
            # 3 = Loaded, 2 = Loading, 1 = NotLoad, 0 = NotExist
            state = utility.load_state(name)
            
            # Robust check for different SDK versions
            state_str = str(state)
            
            if state == 3 or "Loaded" in state_str:
                # Get row count if possible
                col = Collection(name)
                count = col.num_entities
                loaded_collections.append((name, count))
                logger.debug(f"Collection '{name}' is loaded with {count} entities")
            elif state == 2 or "Loading" in state_str:
                loading_collections.append(name)
                logger.debug(f"Collection '{name}' is currently loading")
            else:
                unloaded_collections.append(name)
                logger.debug(f"Collection '{name}' is not loaded")
        except Exception as e:
            logger.warning(f"⚠️ Could not check state for collection '{name}': {e}")

    # 4. Generate and display report
    _print_status_report(loaded_collections, unloaded_collections, loading_collections)
    
    # Log summary
    logger.info(
        f"Collection summary: {len(loaded_collections)} loaded, "
        f"{len(unloaded_collections)} unloaded, {len(loading_collections)} loading"
    )

def main() -> None:
    """Main entry point for the script."""
    try:
        inspect_milvus()
    except KeyboardInterrupt:
        logger.info("\n🛑 Inspection cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Unexpected error during inspection: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()