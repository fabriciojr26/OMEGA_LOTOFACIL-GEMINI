FROM python:3.11-slim

# Define o diretório de trabalho principal
WORKDIR /app

# Copia primeiro o arquivo de dependências para aproveitar o cache do Docker
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todos os arquivos do backend para o diretório de trabalho
COPY backend/ .

# Expõe a porta que a aplicação vai usar
ENV PORT=8080 HOST=0.0.0.0

# Comando para iniciar a aplicação
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]

