# Wildlife Tracking Demo - Deployment Guide

This guide explains how to deploy the interactive web demo for your wildlife tracking project.

## Demo Overview

The demo is built with **Streamlit**, a Python framework for creating interactive data applications. It showcases:
- Algorithm demonstrations with pre-computed tracking videos
- Performance metrics (IoU and FPS graphs)
- Technical explanations of each algorithm
- Comparative analysis

## Local Testing

### Prerequisites
- Python 3.11+
- All dependencies from `requirements.txt`

### Run Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser to `http://localhost:8501`

The app will automatically reload when you make changes to `app.py`.

## Deploy to Streamlit Cloud (FREE)

Streamlit Cloud offers free hosting for public GitHub repositories. Perfect for portfolio projects!

### Step-by-Step Deployment

1. **Push to GitHub**
   - Ensure your repository is public on GitHub
   - Make sure `app.py` and `requirements.txt` are committed

2. **Sign up for Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

3. **Deploy New App**
   - Click "New app"
   - Select your repository: `428-project-final`
   - Main file path: `app.py`
   - Click "Deploy"

4. **Wait for Deployment**
   - Initial deployment takes 2-5 minutes
   - Streamlit will install dependencies and start your app

5. **Get Your URL**
   - You'll get a URL like: `https://username-428-project-final.streamlit.app`
   - Share this on LinkedIn!

### Important Notes for Streamlit Cloud

- **Repository must be public** (for free tier)
- **File size limits**: Keep video files under 100MB each
  - If videos are too large, consider compressing them or using shorter clips
  - Alternatively, host videos elsewhere and link them
- **Dependencies**: All packages in `requirements.txt` must be compatible
- **Automatic updates**: Push to GitHub to automatically redeploy

## Alternative Hosting Options

### Option 2: GitHub Pages + External Videos

If video files are too large for Streamlit Cloud:

1. Host videos on YouTube or Vimeo
2. Create a static HTML/JavaScript version
3. Deploy to GitHub Pages (free)

### Option 3: Heroku (Free Tier)

Heroku also offers free hosting with more storage:

1. Create a `Procfile`:
```
web: streamlit run app.py --server.port=$PORT
```

2. Deploy via Heroku CLI:
```bash
heroku create your-app-name
git push heroku main
```

### Option 4: Railway.app

Railway offers free tier with generous limits:

1. Connect your GitHub repository
2. Railway auto-detects Streamlit
3. Deploys automatically

## Optimizing for Deployment

### Reduce Video File Sizes

If your videos are too large, compress them:

```bash
# Install ffmpeg
brew install ffmpeg  # macOS
# or: sudo apt install ffmpeg  # Linux

# Compress videos
ffmpeg -i input.mp4 -vcodec libx264 -crf 28 output.mp4
```

### Use Sample Clips

Instead of full videos, use 10-15 second clips showing key tracking moments:

```bash
# Extract first 15 seconds
ffmpeg -i input.mp4 -t 15 -c copy output.mp4
```

## Troubleshooting

### App won't start on Streamlit Cloud

1. Check logs in Streamlit Cloud dashboard
2. Verify all dependencies are in `requirements.txt`
3. Check Python version compatibility

### Videos not loading

1. Verify video paths in `app.py`
2. Check file sizes (keep under 100MB each)
3. Ensure videos are in the repository

### Slow loading

1. Compress videos further
2. Use lazy loading for images
3. Consider hosting videos externally

## Sharing Your Demo

### For LinkedIn

1. Create a post with:
   - Link to live demo
   - Link to GitHub repository
   - Brief description of technical approach
   - Screenshots of the demo

Example post:
```
Excited to share my wildlife tracking project using classical computer vision! 🦓

Built custom implementations of ICLK, MOSSE, and Mean Shift trackers,
plus Kalman filtering for improved accuracy.

🔗 Live Demo: [your-streamlit-url]
💻 GitHub: [your-github-url]

Tech stack: Python, OpenCV, NumPy, Streamlit
Key finding: MOSSE tracker achieved 60+ FPS with 0.7+ IoU

#ComputerVision #MachineLearning #Python #OpenCV
```

### For Your Resume

Include:
- Link to live demo in project description
- "Interactive web demo available at [URL]"
- Mention technologies: "Deployed with Streamlit Cloud"

## Customization

### Update Your Name

In `app.py`, line 360:
```python
<p>Implemented by Amogh Panhale</p>
```
(Already has your name!)

### Add More Content

Edit `app.py` to add:
- Additional algorithms
- More datasets
- Custom explanations
- Links to papers

### Styling

Modify the CSS in the `st.markdown()` blocks at the top of `app.py`
to change colors, fonts, and layout.

## Support

If you encounter issues:
1. Check Streamlit documentation: https://docs.streamlit.io
2. Streamlit community forum: https://discuss.streamlit.io
3. File issues on this repository

## License

This demo uses the same license as the main project. Please cite appropriately
if using this code.
