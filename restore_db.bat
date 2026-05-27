@echo off
echo === Clinical NLP Database Restore ===
echo.

echo Step 1: Pulling pgvector image...
docker pull pgvector/pgvector:pg16

echo.
echo Step 2: Starting fresh container...
docker run -d ^
  --name clinical-nlp-pg ^
  -e POSTGRES_USER=postgres ^
  -e POSTGRES_PASSWORD=postgres ^
  -e POSTGRES_DB=clinical_nlp ^
  -p 5432:5432 ^
  -v clinical_nlp_pgdata:/var/lib/postgresql/data ^
  pgvector/pgvector:pg16

echo.
echo Step 3: Waiting for PostgreSQL to be ready...
timeout /t 15

echo.
echo Step 4: Copying dump into container...
docker cp clinical_nlp_backup.dump clinical-nlp-pg:/tmp/clinical_nlp_backup.dump

echo.
echo Step 5: Restoring database...
docker exec clinical-nlp-pg pg_restore -U postgres -d clinical_nlp -Fc /tmp/clinical_nlp_backup.dump

echo.
echo Step 6: Verifying...
docker exec clinical-nlp-pg psql -U postgres -d clinical_nlp -c "SELECT COUNT(*) FROM clinical_nlp.concepts;"
docker exec clinical-nlp-pg psql -U postgres -d clinical_nlp -c "SELECT COUNT(*) FROM clinical_nlp.embeddings;"

echo.
echo === Done! Database ready at localhost:5432 ===
echo Expected: 90904 concepts and embeddings
pause