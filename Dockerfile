FROM node:22-bookworm-slim AS build

WORKDIR /app

ENV VITE_BASE_PATH=/
ENV VITE_STORAGE_ENGINE=localStorage
ENV VITE_REPO_URL=https://github.com/revisit-studies/study

COPY package.json yarn.lock ./
RUN corepack enable && yarn install --frozen-lockfile

COPY . .
RUN corepack yarn build

FROM nginx:1.27-alpine

COPY docker/nginx/default.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
