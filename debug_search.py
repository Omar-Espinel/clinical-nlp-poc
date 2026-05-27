import asyncio
import sys

# Add src to path
sys.path.insert(0, '.')

from src.snomed_search.pgvector_cascade import PgVectorCascadeStrategy

async def test():
    print("=" * 60)
    print("Initializing PgVectorCascadeStrategy...")
    print("=" * 60)
    
    strategy = PgVectorCascadeStrategy(
        dsn="postgresql://postgres:postgres@localhost:5432/clinical_nlp"
    )
    
    # Test exact search for "cancer"
    print("\n" + "=" * 60)
    print("TEST: search('cancer')")
    print("=" * 60)
    
    results = await strategy.search(query="cancer", limit=10)
    
    print(f"\nResults returned: {len(results)}")
    
    if results:
        print("\n✓ Search worked!")
        for i, result in enumerate(results[:5], 1):
            print(f"\n  [{i}] {result.get('preferred_term', 'N/A')}")
            print(f"      Strategy: {result.get('strategy', 'N/A')}")
            print(f"      Confidence: {result.get('confidence', 'N/A')}")
    else:
        print("\n✗ Search returned ZERO results!")
        print("   This is the bug.")
    
    # Test get_top_neighbors directly
    print("\n" + "=" * 60)
    print("TEST: get_top_neighbors('cancer', min_confidence=0.0)")
    print("=" * 60)
    
    neighbors = await strategy.get_top_neighbors(
        query="cancer", 
        limit=10, 
        min_confidence=0.0  # Even with 0 confidence
    )
    
    print(f"\nNeighbors returned: {len(neighbors)}")
    if neighbors:
        print("✓ get_top_neighbors returned results:")
        for n in neighbors[:5]:
            print(f"  - {n}")
    else:
        print("✗ get_top_neighbors returned ZERO results!")

if __name__ == "__main__":
    asyncio.run(test())