docker login
docker build -t fadyadel2382001/ds-toolkit:latest .
docker run -d -p 8000:8000 -e API_KEY=12345 fadyadel2382001/ds-toolkit:latest
docker push fadyadel2382001/ds-toolkit:latest