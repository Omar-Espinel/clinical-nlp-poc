@echo off
echo === Clinical NLP Database Restore ===
echo.

echo Checking that dump file exists...
if not exist clinical_nlp_backup.dump (
    echo ERROR: clinical_nlp_backup.dump not found in this folder.
    echo Download it from SharePoint and place it here first.
    pause
    exit /b 1
)

echo.
echo Step 1: Pulling pgvector image...
docker pull pgvector/pgvector:pg16

echo.
echo Step 2: Checking if container already exists...
docker rm -f clinical-nlp-pg 2>nul
echo Starting fresh container...
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
:waitloop
docker exec clinical-nlp-pg pg_isready -U postgres >nul 2>&1
if errorlevel 1 (
    timeout /t 3 >nul
    goto waitloop
)
echo PostgreSQL is ready.

echo.
echo Step 4: Copying dump into container...
docker cp clinical_nlp_backup.dump clinical-nlp-pg:/tmp/clinical_nlp_backup.dump

echo.
echo Step 5: Restoring database...
docker exec clinical-nlp-pg pg_restore -U postgres -d clinical_nlp -Fc /tmp/clinical_nlp_backup.dump
if errorlevel 1 (
    echo ERROR: Restore failed.
    pause
    exit /b 1
)

echo.
echo Step 6: Verifying...
docker exec clinical-nlp-pg psql -U postgres -d clinical_nlp -c "SELECT COUNT(*) FROM clinical_nlp.concepts;"
docker exec clinical-nlp-pg psql -U postgres -d clinical_nlp -c "SELECT COUNT(*) FROM clinical_nlp.embeddings;"
docker exec clinical-nlp-pg psql -U postgres -d clinical_nlp -c "SELECT COUNT(*) FROM clinical_nlp.synonyms;"

echo.
echo === Done! ===
echo Expected: 90904 for all three counts
echo Database ready at localhost:5432
pause