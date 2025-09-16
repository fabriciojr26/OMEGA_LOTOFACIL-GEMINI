FROM python:3.11-slim

# Define o diretório de trabalho onde a aplicação será executada
WORKDIR /app

# Copia o arquivo de dependências para o diretório de trabalho.
# Fazer isso separadamente aproveita o cache do Docker.
COPY requirements.txt .

# Instala as dependências listadas no requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação (apenas o server.py) para o diretório de trabalho
COPY server.py .

# Define as variáveis de ambiente para o servidor Uvicorn
ENV PORT=8080
ENV HOST=0.0.0.0

# Comando que será executado quando o container iniciar.
# Inicia o servidor FastAPI com Uvicorn.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
