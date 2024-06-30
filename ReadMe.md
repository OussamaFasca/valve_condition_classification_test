to build the docker image :
```
docker build . -t valve-condition-classifier
```
to run the image
```
docker run -p 8000:8000 -t valve-condition-classifier
```
to test the api using curl :
```
curl  -X POST \
  'http://0.0.0.0:8000/predict' \
  --header 'Accept: */*' \
  --header 'Content-Type: application/json' \
  --data-raw '{
  "cycle_number" : 2053
}'
```