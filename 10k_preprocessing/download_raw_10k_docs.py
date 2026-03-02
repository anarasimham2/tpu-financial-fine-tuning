import os
import argparse
from edgar import set_identity, Company
from tqdm import tqdm

def download_10ks(tickers, output_dir, identity):
    set_identity(identity)
    os.makedirs(output_dir, exist_ok=True)

    for ticker in tqdm(tickers, desc="Downloading 10-Ks", unit="file"):
        file_path = os.path.join(output_dir, f"{ticker}_10k.md")
        
        if os.path.exists(file_path):
            continue 

        try:
            company = Company(ticker)
            filings = company.get_filings(form="10-K")
            
            if filings:
                filing = filings.latest()
                markdown_content = filing.markdown()
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(markdown_content)
            else:
                print(f"No 10-K found for {ticker}")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download 10-K filings from SEC EDGAR")
    parser.add_argument("--tickers", nargs="+", help="Tickers to download (space separated)")
    parser.add_argument("--output_dir", default="input_data/raw", help="Directory to save files")
    parser.add_argument("--identity", default="<Your identity>", help="User agent identity for SEC")
    
    args = parser.parse_args()
    
    # Default to a small set if none provided
    tickers = args.tickers if args.tickers else ["GOOGL"]
    download_10ks(tickers, args.output_dir, args.identity)
