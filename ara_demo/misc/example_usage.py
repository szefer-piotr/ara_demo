#!/usr/bin/env python3
"""
Example usage of the BioRxivSearchBuilder class

This script demonstrates how to use the BioRxivSearchBuilder class
programmatically to generate search URLs and open them in the browser.
"""

from search_biorxiv import BiorxivSearchBuilder, BiorxivSearchOptions


def main():
    """Example of programmatic usage"""
    
    # Create a search builder instance
    builder = BiorxivSearchBuilder()
    
    # Example 1: Basic search for population ecology
    print("=== Example 1: Population Ecology Search ===")
    opts1 = BiorxivSearchOptions(
        terms="population ecology",
        subjects=["Evolutionary Biology", "Ecology"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        numresults=5
    )
    
    url1 = builder.build_search_url(opts1)
    print(f"Generated URL: {url1}")
    
    # Example 2: Machine learning in bioinformatics
    print("\n=== Example 2: Machine Learning in Bioinformatics ===")
    opts2 = BiorxivSearchOptions(
        terms="machine learning, bioinformatics",
        subjects=["Bioinformatics", "Computational Biology"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        numresults=10,
        sort="date-desc"  # Most recent first
    )
    
    url2 = builder.build_search_url(opts2)
    print(f"Generated URL: {url2}")
    
    # Example 3: CRISPR research with custom encoding
    print("\n=== Example 3: CRISPR Research with Custom Encoding ===")
    opts3 = BiorxivSearchOptions(
        terms="CRISPR, gene editing",
        subjects=["Genetics", "Molecular Biology"],
        start_date="2023-01-01",
        end_date="2024-12-31",
        numresults=15,
        use_plus_in_terms=False,  # Use spaces instead of plus signs
        double_encode_commas=True  # Double-encode commas
    )
    
    url3 = builder.build_search_url(opts3)
    print(f"Generated URL: {url3}")
    
    # Example 4: Show search configuration
    print("\n=== Example 4: Search Configuration Display ===")
    builder.print_search_info(opts1)
    
    # Example 5: Generate URLs for different subjects
    print("\n=== Example 5: Multiple Subject Searches ===")
    subjects_list = [
        ["Neuroscience"],
        ["Cancer Biology"],
        ["Microbiology"],
        ["Plant Biology"]
    ]
    
    for subjects in subjects_list:
        opts = BiorxivSearchOptions(
            terms="machine learning",
            subjects=subjects,
            start_date="2024-01-01",
            end_date="2024-12-31",
            numresults=5
        )
        url = builder.build_search_url(opts)
        print(f"{', '.join(subjects)}: {url[:80]}...")


if __name__ == "__main__":
    main()
