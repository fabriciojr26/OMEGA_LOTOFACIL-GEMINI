FROM python:3.11-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Como vamos dizer ao Coyeb para usar 'backend' como Root Directory,
# os caminhos aqui são simples e diretos.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código do diretório 'backend' para /app
COPY . .

# Expõe a porta e define o host
ENV PORT=8080
ENV HOST=0.0.0.0

# Comando para iniciar o servidor
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
