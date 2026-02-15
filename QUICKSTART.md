# Quick Start - Web Demo

Get your interactive wildlife tracking demo running in 5 minutes!

## What You're Getting

An interactive web application that showcases:
- 🎥 Live tracking demonstrations on deer and zebra datasets
- 📊 Performance metrics (IoU and FPS graphs)
- 🧠 Technical explanations of each algorithm
- 📈 Comparative analysis of all trackers

## Run Locally (2 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py

# 3. Open browser to http://localhost:8501
```

That's it! The demo is now running on your machine.

## Deploy to Web (5 minutes)

### Option 1: Streamlit Cloud (Recommended - FREE)

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Add web demo"
   git push origin main
   ```

2. **Go to [share.streamlit.io](https://share.streamlit.io)**
   - Sign in with GitHub
   - Click "New app"
   - Select this repository
   - Set main file: `app.py`
   - Click "Deploy"

3. **Get your URL**
   - You'll get: `https://[username]-428-project-final.streamlit.app`
   - Share on LinkedIn!

**Important**: Repository must be public for free hosting.

### Option 2: Railway (Also FREE)

1. Go to [railway.app](https://railway.app)
2. Click "Start a New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway auto-detects and deploys

## For LinkedIn

Once deployed, create a post like:

```
Excited to share my Computer Vision project for wildlife tracking! 🦓

I implemented and compared classical tracking algorithms (ICLK, MOSSE, Mean Shift)
with Kalman filtering on real animal datasets.

🔗 Try the live demo: [your-url-here]
💻 Source code: github.com/[your-username]/428-project-final

Key results:
- MOSSE tracker: 60+ FPS, 0.7+ IoU
- Custom ICLK implementation from research papers
- Full performance comparison across algorithms

Tech: Python, OpenCV, NumPy, Streamlit

#ComputerVision #OpenCV #Python #MachineLearning
```

## Customization

Want to personalize? Edit [app.py](app.py):

- **Line 360**: Your name (already set to "Amogh Panhale")
- **Lines 11-30**: Colors and styling (CSS)
- **All pages**: Add more content, explanations, or datasets

## Troubleshooting

**Videos not showing?**
- Check that output videos exist in `outputs/` directory
- Paths are case-sensitive

**App won't start?**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Deployment fails?**
- Check [DEMO_DEPLOYMENT.md](DEMO_DEPLOYMENT.md) for detailed troubleshooting
- Verify all files are committed to Git

## File Structure

```
app.py                  # Main Streamlit application
requirements.txt        # Python dependencies
outputs/                # Pre-computed tracking results
  output_deer/          # Deer dataset results
  output_zebra/         # Zebra dataset results
```

## Next Steps

1. **Test locally** to make sure everything works
2. **Compress videos** if too large (see DEMO_DEPLOYMENT.md)
3. **Deploy to Streamlit Cloud**
4. **Share on LinkedIn** with your unique URL
5. **Add to resume** under projects section

## Pro Tips

✅ **Add screenshots** to your LinkedIn post
✅ **Tag skills**: Computer Vision, Python, OpenCV
✅ **Explain the impact**: Wildlife monitoring, conservation tech
✅ **Show metrics**: "60+ FPS", "0.7 IoU", "3 algorithms"
✅ **Link both**: Live demo AND GitHub repo

## Support

- Full deployment guide: [DEMO_DEPLOYMENT.md](DEMO_DEPLOYMENT.md)
- Project README: [README.md](README.md)
- Streamlit docs: https://docs.streamlit.io

---

**Ready to impress recruiters?** Deploy now and share your work! 🚀
