# Build stage
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY services/frontend/package*.json ./

# Install dependencies
RUN npm install

# Build args for React env vars (baked in at build time)
ARG REACT_APP_AUTH_API_URL
ARG REACT_APP_CHATBOT_API_URL
ENV REACT_APP_AUTH_API_URL=$REACT_APP_AUTH_API_URL
ENV REACT_APP_CHATBOT_API_URL=$REACT_APP_CHATBOT_API_URL

# Copy source code
COPY services/frontend/public ./public
COPY services/frontend/src ./src

# Build the React app
RUN npm run build

# Production stage
FROM node:18-alpine

WORKDIR /app

# Install serve to run the React app
RUN npm install -g serve

# Copy built app from builder
COPY --from=builder /app/build ./build

# Expose port
EXPOSE 5007

# Start the app
CMD ["serve", "-s", "build", "-l", "5007"]
