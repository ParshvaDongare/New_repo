# Railway Deployment Guide

This guide covers deploying the Policy Intelligence API to Railway with a 4GB build limit.

## Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **GitHub Repository**: Your code should be in a GitHub repository
3. **Environment Variables**: Prepare your environment variables

## Environment Variables

Set these environment variables in Railway:

### Required Variables
```
DATABASE_URL=postgresql://username:password@host:port/database
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=gcp-starter
BEARER_TOKEN=your_bearer_token
```

### API Keys (Multiple for Parallel Processing)
```
GROQ_API_KEYS_1=your_groq_api_key_1
GROQ_API_KEYS_2=your_groq_api_key_2
GROQ_API_KEYS_3=your_groq_api_key_3
GROQ_API_KEYS_4=your_groq_api_key_4
GROQ_API_KEYS_5=your_groq_api_key_5

JINA_API_KEY_1=your_jina_api_key_1
JINA_API_KEY_2=your_jina_api_key_2
JINA_API_KEY_3=your_jina_api_key_3
JINA_API_KEY_4=your_jina_api_key_4
JINA_API_KEY_5=your_jina_api_key_5
JINA_API_KEY_6=your_jina_api_key_6
```

### Optional Configuration
```
DEBUG=false
LOG_LEVEL=INFO
MAX_MEMORY_MB=400
ENABLE_OCR=false
ENABLE_ON_DEMAND=true
```

## Deployment Steps

### 1. Connect GitHub Repository

1. Go to Railway Dashboard
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository
5. Select the main branch

### 2. Configure Build Settings

Railway will automatically detect the configuration from:
- `railway.json` - Deployment configuration
- `nixpacks.toml` - Build configuration
- `requirements.txt` - Python dependencies

### 3. Set Environment Variables

1. Go to your project in Railway
2. Navigate to "Variables" tab
3. Add all required environment variables
4. Save changes

### 4. Deploy

1. Railway will automatically start building
2. Monitor the build logs for any issues
3. The app will be available at the provided URL

## Build Optimizations

This deployment is optimized for Railway's 4GB build limit:

### Removed Heavy Dependencies
- `opencv-python` (replaced with lighter alternatives)
- `pytesseract` (replaced with `pytesseract-headless`)
- `tabula-py` (removed - requires Java)

### Lightweight Alternatives
- `fitz` instead of `PyMuPDF` for PDF processing
- `pdfplumber` for table extraction
- `pytesseract-headless` for OCR

### Build Configuration
- Uses `--no-cache-dir` for pip installs
- Includes system dependencies in nixpacks
- Optimized uvicorn settings for Railway

## Monitoring

### Health Check
The app includes a health check endpoint at `/health` that Railway uses to monitor the service.

### Logs
Monitor logs in Railway dashboard for:
- Build errors
- Runtime errors
- Performance issues

### Memory Usage
The app is configured to stay under 400MB to leave headroom for Railway's 512MB limit.

## Troubleshooting

### Build Failures
1. Check build logs for dependency issues
2. Verify all environment variables are set
3. Ensure repository is public or Railway has access

### Runtime Errors
1. Check application logs
2. Verify database connection
3. Ensure API keys are valid

### Performance Issues
1. Monitor memory usage
2. Check concurrent request limits
3. Verify rate limiting settings

## Scaling

Railway automatically scales based on traffic. The app is configured with:
- Single worker process
- Request limits to prevent memory issues
- Health checks for reliability

## Cost Optimization

- Use Railway's free tier for development
- Monitor usage to stay within limits
- Consider paid plans for production workloads

## Security

- All API keys are stored as environment variables
- Bearer token authentication required
- HTTPS enabled by default
- No sensitive data in code 