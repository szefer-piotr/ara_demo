#!/usr/bin/env python3
"""
Demo script showcasing the complete bioarxiv pipeline.
This script demonstrates how the pipeline works from downloading papers
to performing similarity search on user queries.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def demo_pipeline():
    """Demonstrate the complete pipeline workflow."""
    
    print("🚀 Bioarxiv Pipeline Demo")
    print("=" * 50)
    
    # Step 1: Paper Download Simulation
    print("\n📥 Step 1: Paper Download")
    print("-" * 30)
    
    # Simulate papers that would be downloaded
    sample_papers = [
        {
            "doi": "10.1101/2023.01.01.123456",
            "title": "Deep Learning Approaches in Genomic Sequence Analysis",
            "server": "biorxiv",
            "content": """
            Deep learning has revolutionized genomic sequence analysis in recent years. 
            Convolutional neural networks and recurrent neural networks have shown remarkable 
            performance in predicting gene function and regulatory elements. The application 
            of transformer architectures to genomic data has opened new possibilities for 
            understanding complex biological relationships.
            """
        },
        {
            "doi": "10.1101/2023.01.02.789012", 
            "title": "Machine Learning for Protein Structure Prediction: A Comprehensive Review",
            "server": "biorxiv",
            "content": """
            Machine learning approaches have significantly advanced protein structure prediction. 
            AlphaFold2 demonstrated unprecedented accuracy in predicting protein structures 
            from amino acid sequences. This breakthrough has implications for drug discovery, 
            understanding disease mechanisms, and protein engineering.
            """
        },
        {
            "doi": "10.1101/2023.01.03.345678",
            "title": "Transformer Models in Bioinformatics: Applications and Challenges",
            "server": "biorxiv", 
            "content": """
            Transformer models have shown exceptional performance in various bioinformatics tasks. 
            BERT-based models trained on protein sequences can predict protein function and 
            structure. Attention mechanisms allow these models to capture long-range dependencies 
            in biological sequences.
            """
        }
    ]
    
    print(f"📚 Downloaded {len(sample_papers)} papers:")
    for i, paper in enumerate(sample_papers, 1):
        print(f"   {i}. {paper['title']}")
        print(f"      DOI: {paper['doi']}")
        print(f"      Server: {paper['server']}")
    
    # Step 2: Text Chunking Simulation
    print("\n✂️  Step 2: Text Chunking")
    print("-" * 30)
    
    try:
        from chunk_arxiv_papers import chunk_text
        
        all_chunks = []
        chunk_metadata = []
        
        for paper in sample_papers:
            chunks = chunk_text(paper['content'], size=150, overlap=30)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "paper_id": paper['doi'].replace("/", "_"),
                    "paper_title": paper['title'],
                    "chunk_id": i,
                    "doi": paper['doi'],
                    "source": f"{paper['doi'].replace('/', '_')}.pdf"
                })
        
        print(f"📝 Created {len(all_chunks)} chunks from {len(sample_papers)} papers")
        print(f"   Average chunks per paper: {len(all_chunks) / len(sample_papers):.1f}")
        
        # Show sample chunks
        print("\n   Sample chunks:")
        for i, chunk in enumerate(all_chunks[:3]):
            print(f"   Chunk {i+1}: {chunk[:80]}...")
            
    except ImportError as e:
        print(f"⚠️  Chunking module not available: {e}")
        print("   Using mock chunks for demo...")
        
        # Mock chunks for demo
        all_chunks = [
            "Deep learning has revolutionized genomic sequence analysis in recent years.",
            "Machine learning approaches have significantly advanced protein structure prediction.",
            "Transformer models have shown exceptional performance in various bioinformatics tasks."
        ]
        chunk_metadata = [
            {"paper_id": "paper_1", "paper_title": "Sample Paper 1", "chunk_id": 0, "doi": "10.1101/2023.01.01.123456"},
            {"paper_id": "paper_2", "paper_title": "Sample Paper 2", "chunk_id": 0, "doi": "10.1101/2023.01.02.789012"},
            {"paper_id": "paper_3", "paper_title": "Sample Paper 3", "chunk_id": 0, "doi": "10.1101/2023.01.03.345678"}
        ]
    
    # Step 3: Chunk Quality Evaluation
    print("\n🔍 Step 3: Chunk Quality Evaluation")
    print("-" * 30)
    
    try:
        from chunk_evaluation import quality_metrics
        
        chunk_qualities = []
        for chunk_text in all_chunks:
            quality = quality_metrics(chunk_text)
            chunk_qualities.append(quality)
        
        print(f"📊 Evaluated quality for {len(chunk_qualities)} chunks")
        
        # Show quality metrics for first chunk
        if chunk_qualities:
            first_quality = chunk_qualities[0]
            print(f"   First chunk metrics:")
            print(f"     Characters: {first_quality['char_len']}")
            print(f"     Tokens: {first_quality['token_est']}")
            print(f"     Lines: {first_quality['lines']}")
            print(f"     ASCII %: {first_quality['pct_ascii']:.1%}")
            print(f"     Alphanumeric %: {first_quality['pct_alnum']:.1%}")
            
    except ImportError as e:
        print(f"⚠️  Evaluation module not available: {e}")
        print("   Skipping quality evaluation...")
    
    # Step 4: Vector Storage Simulation
    print("\n💾 Step 4: Vector Storage")
    print("-" * 30)
    
    print("🗄️  Storing chunks in Qdrant vector database...")
    print(f"   Collection: bioarxiv_chunks")
    print(f"   Total points: {len(all_chunks)}")
    print(f"   Vector dimensions: 1536 (text-embedding-3-small)")
    
    # Simulate storage
    for i, (chunk, metadata) in enumerate(zip(all_chunks, chunk_metadata)):
        print(f"   Point {i+1}: {metadata['paper_title'][:40]}...")
    
    # Step 5: User Query and Search
    print("\n🔍 Step 5: User Query and Search")
    print("-" * 30)
    
    # Example user queries
    user_queries = [
        "What are the latest advances in machine learning for bioinformatics?",
        "How has machine learning improved protein structure prediction?",
        "What are the applications of transformer models in bioinformatics?"
    ]
    
    print("👤 User queries:")
    for i, query in enumerate(user_queries, 1):
        print(f"   {i}. {query}")
    
    # Simulate search results
    print("\n📋 Search results for query 1:")
    print("   Query: What are the latest advances in machine learning for bioinformatics?")
    
    # Mock search results
    search_results = [
        {
            "score": 0.95,
            "text": all_chunks[0],
            "paper_title": chunk_metadata[0]["paper_title"],
            "doi": chunk_metadata[0]["doi"]
        },
        {
            "score": 0.92,
            "text": all_chunks[1], 
            "paper_title": chunk_metadata[1]["paper_title"],
            "doi": chunk_metadata[1]["doi"]
        },
        {
            "score": 0.88,
            "text": all_chunks[2],
            "paper_title": chunk_metadata[2]["paper_title"], 
            "doi": chunk_metadata[2]["doi"]
        }
    ]
    
    for i, result in enumerate(search_results, 1):
        print(f"\n   Result {i} (Score: {result['score']:.2f}):")
        print(f"     Paper: {result['paper_title']}")
        print(f"     DOI: {result['doi']}")
        print(f"     Text: {result['text'][:100]}...")
    
    # Step 6: Pipeline Summary
    print("\n📈 Pipeline Summary")
    print("-" * 30)
    
    print(f"✅ Papers downloaded: {len(sample_papers)}")
    print(f"✅ Chunks created: {len(all_chunks)}")
    print(f"✅ Quality evaluated: {len(chunk_qualities) if 'chunk_qualities' in locals() else 'N/A'}")
    print(f"✅ Vectors stored: {len(all_chunks)}")
    print(f"✅ Search queries processed: {len(user_queries)}")
    print(f"✅ Results returned: {len(search_results)}")
    
    print("\n🎯 Pipeline successfully demonstrated!")
    print("   The system can now answer questions about bioinformatics papers")
    print("   using semantic similarity search over chunked content.")


def demo_search_workflow():
    """Demonstrate the search workflow specifically."""
    
    print("\n🔍 Search Workflow Demo")
    print("=" * 50)
    
    # Simulate a user asking a question
    user_question = "What are the latest advances in machine learning for bioinformatics?"
    
    print(f"👤 User Question: {user_question}")
    print("\n🔄 Processing workflow:")
    
    # Step 1: Query understanding
    print("   1. 📝 Parse and understand user query")
    print(f"      Query: '{user_question}'")
    print("      Keywords: machine learning, bioinformatics, advances")
    
    # Step 2: Embedding generation
    print("   2. 🧠 Generate query embedding")
    print("      Model: text-embedding-3-small")
    print("      Dimensions: 1536")
    print("      Cost: ~$0.00002 per query")
    
    # Step 3: Vector search
    print("   3. 🔍 Perform vector similarity search")
    print("      Database: Qdrant")
    print("      Collection: bioarxiv_chunks")
    print("      Search method: Cosine similarity")
    print("      Top-k: 5 results")
    
    # Step 4: Result ranking
    print("   4. 📊 Rank and filter results")
    print("      Score threshold: 0.7")
    print("      Relevance ranking: Semantic similarity")
    
    # Step 5: Response generation
    print("   5. 💬 Generate user response")
    print("      Format: Ranked list with excerpts")
    print("      Include: Paper title, DOI, relevant text")
    
    print("\n✅ Search workflow completed!")
    print("   User receives relevant paper excerpts with metadata")


def main():
    """Main demo function."""
    
    print("🧬 Bioarxiv Pipeline Demonstration")
    print("=" * 60)
    print("This demo shows how the pipeline processes papers and enables")
    print("semantic search over bioinformatics literature.")
    print()
    
    try:
        # Run main pipeline demo
        demo_pipeline()
        
        # Run search workflow demo
        demo_search_workflow()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("\nTo run the actual pipeline:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Set up MinIO and Qdrant")
        print("   3. Configure OpenAI API key")
        print("   4. Run: python download_biorxiv_papers.py --server biorxiv")
        print("   5. Run: python chunk_arxiv_papers.py [options]")
        print("   6. Run: python similarity_search_arxiv_papers.py --query 'your query'")
        print("\nTo run tests:")
        print("   make test")
        print("   python tests/run_tests.py")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("   This is expected if dependencies are not installed.")
        print("   The demo shows the conceptual workflow.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
