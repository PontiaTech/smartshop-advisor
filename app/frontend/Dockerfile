# Etapa 1: Build con Node
FROM node:20-alpine AS build

WORKDIR /app

# Copiamos dependencias primero para aprovechar la cache de Docker
COPY frontend/package*.json ./
RUN npm install

# Copiamos el resto del código
COPY frontend/ .

# Compilamos para producción (crea /app/dist o /app/build según framework)
RUN npm run build

# Etapa 2: Servir con Nginx
FROM nginx:alpine

# Copiamos el build del frontend a la carpeta pública de Nginx
COPY --from=build /app/dist /usr/share/nginx/html

# Exponer puerto 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
