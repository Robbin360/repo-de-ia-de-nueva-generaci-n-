FROM node:22-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
    && python3 -m venv /opt/aethel-venv \
    && /opt/aethel-venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0+cpu \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/aethel-venv/bin:${PATH}"
WORKDIR /app
COPY . .
RUN npm install -g corepack@latest && corepack pnpm install && corepack pnpm run build

ENV NODE_ENV=production
CMD ["node", "dist/index.js"]
