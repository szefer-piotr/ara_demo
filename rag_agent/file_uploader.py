from minio import Minio
from minio.error import S3Error
from pathlib import Path

def main():
    client = Minio("localhost:9000",
                   access_key="minioadmin",
                   secret_key="minioadmin",
                   secure = False
                   )
    
    chunk_dir = Path("rag_agent/preprocessed")

    bucket_name = "arxiv-chunks"

    found = client.bucket_exists(bucket_name)

    print(found)

    if not found:
        client.make_bucket(bucket_name)
        print("Created bucket", bucket_name)
    else:
        print("Bucket", bucket_name, "aleready exists")

    # Upload files
    print(f"Uploading files ...")
    print(chunk_dir.glob("*_section.json"))

    for chunk_file in chunk_dir.glob("*_section.json"):
        print(chunk_file.name)
        client.fput_object(
            bucket_name=bucket_name,
            object_name=chunk_file.name,
            file_path=str(chunk_file),
        )
        print(f"Uploaded: {chunk_file.name}")

if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occured", exc)