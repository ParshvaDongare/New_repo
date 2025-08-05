# Railway Deployment Optimizations

This document summarizes all optimizations made to prepare the application for Railway deployment with a 4GB build limit.

## Build Optimizations

### 1. Dependencies Optimization

**Removed Heavy Dependencies:**
- `opencv-python==4.8.1.78` (400MB+) - Replaced with lighter alternatives
- `pytesseract==0.3.10` - Replaced with `pytesseract-headless`
- `tabula-py==2.8.2` - Removed (requires Java, too heavy)
- `PyMuPDF==1.23.8` - Replaced with `fitz==0.0.1.dev2`

**Added Lightweight Alternatives:**
- `pdfplumber==0.10.3` - For table extraction
- `pytesseract-headless==0.3.10` - For OCR without heavy dependencies
- `fitz==0.0.1.dev2` - Lighter PDF processing

### 2. Build Configuration

**nixpacks.toml:**
- Added system dependencies: `tesseract`, `poppler_utils`
- Used `--no-cache-dir` for pip installs
- Added build verification step
- Optimized uvicorn settings

**railway.json:**
- Added health check interval
- Optimized start command with request limits
- Added restart policies

### 3. Memory Management

**Configuration:**
- Memory limit set to 400MB (leaving 112MB headroom)
- Request limits: 1000 max requests, 10 concurrent
- Single worker process for stability

**Code Optimizations:**
- Graceful handling of missing OCR dependencies
- Memory monitoring enabled
- Cache size limits

## Deployment Files

### 1. Configuration Files
- ✅ `railway.json` - Railway deployment configuration
- ✅ `nixpacks.toml` - Build configuration
- ✅ `Procfile` - Process definition
- ✅ `runtime.txt` - Python version specification

### 2. Documentation
- ✅ `DEPLOYMENT.md` - Detailed deployment guide
- ✅ `RAILWAY_OPTIMIZATIONS.md` - This optimization summary
- ✅ Updated `README.md` with deployment instructions

### 3. Automation
- ✅ `.github/workflows/deploy.yml` - GitHub Actions workflow
- ✅ `.dockerignore` - Build context optimization
- ✅ `scripts/health_check.py` - Deployment verification

## Environment Variables

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

## Code Changes

### 1. Parser Service
- Updated OCR availability check to remove OpenCV dependency
- Graceful handling of missing heavy dependencies
- Maintained functionality with lighter alternatives

### 2. Configuration
- Memory limits optimized for Railway
- Request limits to prevent memory issues
- Health check endpoints configured

## Deployment Process

### 1. GitHub Integration
1. Connect repository to Railway
2. Set environment variables
3. Automatic deployment on push to main branch

### 2. Build Process
1. Railway detects Python project
2. Uses nixpacks for build
3. Installs dependencies with optimizations
4. Verifies build with test imports

### 3. Runtime
1. Single worker process
2. Request limits for stability
3. Health checks for monitoring
4. Automatic restarts on failure

## Monitoring and Maintenance

### 1. Health Checks
- `/health` endpoint for Railway monitoring
- `/debug/index-stats` for service status
- Health check script for verification

### 2. Logs
- Application logs in Railway dashboard
- Build logs for troubleshooting
- Error tracking and monitoring

### 3. Performance
- Memory usage monitoring
- Request rate limiting
- Concurrent connection limits

## Cost Optimization

### 1. Free Tier Usage
- Optimized for Railway's free tier
- Memory usage under 512MB limit
- Efficient resource utilization

### 2. Scaling Considerations
- Single worker for cost efficiency
- Request limits to prevent overuse
- Monitoring for usage optimization

## Security

### 1. Environment Variables
- All sensitive data in environment variables
- No hardcoded secrets
- Secure API key management

### 2. Authentication
- Bearer token authentication
- HTTPS enabled by default
- Secure API endpoints

## Troubleshooting

### Common Issues
1. **Build Failures**: Check dependency conflicts
2. **Memory Issues**: Monitor usage, adjust limits
3. **API Errors**: Verify environment variables
4. **Performance**: Check request limits and concurrency

### Debugging
1. Check Railway logs
2. Use health check script
3. Monitor memory usage
4. Verify environment variables

## Success Metrics

### Build Success
- ✅ Under 4GB build limit
- ✅ All dependencies install correctly
- ✅ Application starts successfully

### Runtime Success
- ✅ Memory usage under 400MB
- ✅ Health checks pass
- ✅ API endpoints respond correctly
- ✅ No dependency errors

### Performance
- ✅ Fast startup time
- ✅ Stable request handling
- ✅ Proper error handling
- ✅ Monitoring and logging 