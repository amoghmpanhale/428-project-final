# Deployment Guide - Quick Reference

## Pre-Deployment Checklist

- [x] Videos converted to web format (`python convert_videos_for_web.py`)
- [x] App tested locally (`streamlit run app.py`)
- [ ] Code pushed to GitHub
- [ ] Repository is public (required for free hosting)

## Deploy to Streamlit Cloud (Recommended)

**Time: 5 minutes | Cost: FREE**

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Add interactive web demo for wildlife tracking"
git push origin main
```

### Step 2: Deploy to Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"Sign in"** → Use GitHub account
3. Click **"New app"**
4. Fill in:
   - **Repository:** `your-username/428-project-final`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy"**

### Step 3: Wait for Deployment

- Initial deployment: 2-5 minutes
- Watch the logs for progress
- App will auto-restart if there are errors

### Step 4: Get Your URL

Your app will be live at:
```
https://[your-username]-428-project-final.streamlit.app
```

### Step 5: Update README

Replace `your-streamlit-url-here` in README.md with your actual URL.

---

## Alternative: Railway.app

**Time: 3 minutes | Cost: FREE (500 hours/month)**

1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select `428-project-final`
5. Railway auto-detects Streamlit and deploys

Your URL: `https://your-app.up.railway.app`

---

## Alternative: Heroku

**Time: 10 minutes | Cost: FREE tier available**

### Additional Files Needed

Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

Create `runtime.txt`:
```
python-3.11.0
```

### Deploy

```bash
# Install Heroku CLI
brew install heroku/brew/heroku  # macOS
# or: curl https://cli-assets.heroku.com/install.sh | sh  # Linux

# Login and deploy
heroku login
heroku create your-app-name
git push heroku main
heroku open
```

---

## Troubleshooting

### Deployment Fails

**Check logs:**
- Streamlit Cloud: Click "Manage app" → "Logs"
- Railway: Click on deployment → "View logs"
- Heroku: `heroku logs --tail`

**Common issues:**

1. **Missing dependencies**
   - Make sure `requirements.txt` is up to date
   - Verify all packages are available on PyPI

2. **Videos too large**
   - Streamlit Cloud limit: 1GB total repo size
   - Check: `du -sh .git`
   - Solution: Compress videos or use shorter clips

3. **Module not found**
   - Check Python version compatibility
   - Update `requirements.txt` versions

### Videos Not Playing

Run the conversion script:
```bash
python convert_videos_for_web.py
```

This re-encodes videos to H.264 (web-compatible).

### App Loads but Shows Errors

1. Check if files exist:
   ```bash
   ls -lh outputs/output_deer/
   ```

2. Verify paths in `app.py` match your directory structure

3. Test locally first:
   ```bash
   streamlit run app.py
   ```

---

## Post-Deployment

### Update Your Live App

Just push to GitHub:
```bash
git add .
git commit -m "Update demo"
git push
```

Streamlit Cloud auto-redeploys on every push.

### Share on LinkedIn

**Sample Post:**

```
Excited to share my Computer Vision project! 🦓

I implemented classical tracking algorithms (ICLK, MOSSE, Mean Shift)
for wildlife monitoring, with full comparative analysis.

🔗 Live Demo: [your-url]
💻 Source: github.com/[username]/428-project-final

Results:
• MOSSE: 60+ FPS, 0.7+ IoU
• Custom implementations from research papers
• Kalman filter integration

Built with Python, OpenCV, NumPy, Streamlit

#ComputerVision #Python #OpenCV #MachineLearning #Wildlife
```

### Add to Resume

**Project Section:**
```
Wildlife Tracking System | Python, OpenCV, Streamlit
• Implemented 3 classical CV tracking algorithms with custom optimizations
• Achieved 60+ FPS with 0.7+ IoU on real-world animal datasets
• Built interactive web demo showcasing algorithm comparisons
• Demo: [your-url] | Code: github.com/[username]/428-project-final
```

---

## Performance Optimization

### Reduce Repository Size

```bash
# Check repo size
du -sh .git

# Remove large files from history (if needed)
git filter-branch --tree-filter 'rm -rf samurai' HEAD
```

### Compress Videos Further

```bash
# Install ffmpeg
brew install ffmpeg

# Compress (example)
ffmpeg -i input.mp4 -c:v libx264 -crf 28 -preset fast output.mp4
```

### Speed Up Loading

1. Use smaller video clips (10-15 seconds)
2. Add lazy loading for images
3. Cache expensive operations with `@st.cache_data`

---

## Cost Comparison

| Platform | Free Tier | Limits | Best For |
|----------|-----------|--------|----------|
| **Streamlit Cloud** | ✅ Unlimited | 1GB repo, public only | Quick demos |
| **Railway** | ✅ 500 hrs/mo | $5 credit/month | More control |
| **Heroku** | ✅ Limited | 550 hrs/month | Production apps |
| **Vercel** | ✅ Yes | Serverless limits | Static sites |

**Recommendation:** Start with **Streamlit Cloud** - it's the easiest and designed for Streamlit apps.

---

## Need Help?

- **Streamlit Docs:** https://docs.streamlit.io
- **Community Forum:** https://discuss.streamlit.io
- **This Project:** See [QUICKSTART.md](QUICKSTART.md) for detailed guide

---

**Ready to deploy? Start with Streamlit Cloud - it takes just 5 minutes!** 🚀
