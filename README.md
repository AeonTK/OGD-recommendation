
## Running the API (Docker)

Build:

`docker build -t ogd .`

Run (loads env vars from `.env`):

`docker run --rm -p 8000:8000 --env-file .env ogd`

### Milvus from inside Docker

If Milvus is running on your host machine:

- Windows/macOS Docker Desktop: set `MILVUS_URI=http://host.docker.internal:19530`
	- You can do this either in `.env` (uncomment the provided line) or override at runtime:
		- `docker run --rm -p 8000:8000 --env-file .env -e MILVUS_URI=http://host.docker.internal:19530 ogd`

On Linux, `host.docker.internal` may require extra configuration (Docker version/daemon setting); the simplest alternative is running Milvus in the same Docker network and using the Milvus container name as the host.

