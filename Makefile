# Add new targets for the improved workflow
biorxiv-direct:
	python3 src/biorxiv_upload_papers.py \
	--num-packages 10 \
	--file-type pdf \
	--month June_2025 \
	-v

biorxiv-all-files:
	python3 src/biorxiv_upload_papers.py \
	--num-packages 10 \
	--file-type all \
	--month June_2025 \
	-v

biorxiv-force:
	python3 src/biorxiv_upload_papers.py \
	--num-packages 10 \
	--force \
	--month June_2025 \
	-v

test-biorxiv-minio:
	python3 src/biorxiv_upload_papers.py --test-minio