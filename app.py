"""
app.py — Samridhi AI v2.0  (M-CADWM & SMIS)
Matches index.html exactly.
"""
from __future__ import annotations
import base64, os, re, time, threading
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from samridhi.config import BASE_DIR, cfg
from samridhi.logger import get_logger
from samridhi.pipeline import Pipeline, RateLimiter
from samridhi.resources import (
    get_analytics, get_expansions, get_feedback_db,
    get_llm, get_vector_db, get_web_cache,
)
from samridhi.tts import autoplay_audio, speak
from samridhi.ui_strings import BRIDGE_JS, UI

log = get_logger()
_PENDING_FEEDBACK_MAX = 20
_FEEDBACK_LAYERS      = frozenset({"faiss","live","fallback","cache"})

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    _HAS_PDF = True
except ImportError:
    _HAS_PDF = False

# ── Logo embedded as base64 ───────────────────────────────────
_LOGO_B64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERETFhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCAD0ARgDASIAAhEBAxEB/8QAHAABAAICAwEAAAAAAAAAAAAAAAYHBQgBAwQC/8QASBAAAQMDAQUEBwQHBAkFAAAAAQACAwQFEQYHEiExQRNRYYEIFCIycZGhI0JSsRUWYnKCosFDY5LRGCQlNFODssLhM0Rzs/D/xAAbAQEAAwEBAQEAAAAAAAAAAAAAAwUGBAIBB//EADMRAAICAQEFBQcEAwEBAAAAAAABAgMEEQUSITFBEyJRcfAGFDJhobHRIySRwTRCgeEz/9oADAMBAAIRAxEAPwDTJERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBERAEWZsGltSX84stjr64ZwXwwOLB8Xch81OLTsL1vVsD60W61tPSoqQ5w8mByjndXX8ckiOy6uv45JFXIrzpPR7lx/ruradh6iCjc/wCpc1e7/R3o3N+z1ZVl3jbOH/2Lne0MZf7o5ntHGT030a/IrzrvR0ubWF1Bqakld0bPSvjz5jeUVu+w/aLb2ufHZ4rhG371HUMeT/CSHfRSwyap/DJE0MmmfwyRWyL3Xiz3azVBp7tbauhl/BUQuYfqF4VOThERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQBF9wRSzzMhgjfLLI4NYxjSXOJ5AAcythdlexujtcUF51nAKm4OAfDbDxjg7jL+J37PIdc8lBkZFePDfmyDIya8eG/Yys9nuyvU2sGMrI42W21E/77VZa137jebz8OHirz0psq0TpwMkNv8A0zWN51FeN5ue9sXujzyVN55G9k6aeWOGGGP2nOIbHEwfRoCpbaNtuhpXyW7RbWzSDLX3KVuWg/3bT/1H5KiWVl7QlpR3Y+P/AL+Ch97y9oS0o7sfH1/Rb18vNrsVCJ7zc6W2UzW+w2RwZkdzGDifIKq9T7eLLSF0OnrRNcXjgJ6pxij8mj2j5kKg7tcrhdq2StudZPWVMhy6SZ5c4/NeRd1GyKIPen3n8zuo2PRW96zvP5ljXnbRruvBZT19PbYz9ykga3+Z2T9VEbhqfU1bIX1t+ukzncTv1TznyysOr+2SQaZ11s9jseobZDNNbpDAKiECOpja7LmOa8DJ6jByD3dVZRrhWu7HTTwO6x1Yte/u6JeCKMFyuIdvC4VYd3iZ2fzWdtO0LXFq3fUdU3WMN5NdUOe35OyFnNquyu7aMJuFI91zsbz7NWxntQno2Vo90+PI/RV2vsZQsjvLiiWEoWxUo8UWLUbZNZ19vfb72+23mmeMFlbRsd5gjGCoBVSsmmdIyBkIJzuMzgfDK6UXpJLgj2klyLD0xs4h1bo99001fYJ7zTZ9atE7dyQjo6N3JwPjhQO4UdXb62airqaWmqYXFksUrS1zHDoQeS77HdrhZLpDcrXUvpqqF2WPb+RHUeCu0PsG1/TZnq2sor/SsDHzMb7UZ5An8cR+beQ8Zqqnb3Y8/uQ23dj3pfD9ihEWT1NY7lp27zWu6QGKeM8COLXtPJzT1ae9YxRtNPRkyaa1QREXw+hERAEREAREQBERAEREAREQBERAEREAXLWuc4NaC5xOAAOJK4V3+jfoNlTJ+u13p2vp4JCy2xPGRJKOchHVren7XwUV90aK3ZPkiK+6NFbsnyRLtiWzePSVBFf71Ax9/qGb0Ubhn1Fh5f8AMPU/dHDnlWFca2jt9vnuVyqmU1JA3fmmeeAH9Seg6rue735ZnhrQC973HgAOJJPctX9tW0GXVt3dbrdI6OxUjyIGA47dw4dq749B0CzFFdm1b3OfCK9cDLU127Vvc58Ir1wPnaztNuGsKl9BQGSisUbvs6cHDpiOT5Mcz3DkF5tl2zO86+bWS0FRDTQ0vAvka5287GcANHQcyoKpls12i33Qs8otzmyUk7szwOJbvcMHDhxBwtXTXXWlFLRI1Cr7KvdqWmhHNQWmssd5qrTXsDaimeWPwcg9xHgRxV3w2HQ9s2AurLpQRC5VFGJWTlg7QyvB3ADzznHDljK9FZQ6N2r2hleJ30lyhYGmdjQZogOTJm/fb0Dh/wCFE/SDrTALLYKfeFJDB2rR0OPs2/INP+Jd3u/ZwlY+K6PzOX3ntZxqXCXVeRUqnuwy8utmt4qJ0m7BcmGmdk8N7mw/4gB5r27Bbxoe032qGs6GGYTsDKaWePfijPHO8MHGeHHHBcbVtFTaWvbNS2CPtbFNMJqeSI7wp3Z3gwkfd/C7kQuaKa7yOm3dmnVLqjZahmFRQ4kY18cjCyWN7Q5rxyIcDwI8FRm2HZD6myo1Do+F8lI3MlVb2gl0A5l0f4meHMeIVxacrG1NL2/usmiZUNyeQe0O/qsnSVRfIZICRuHg/vWWybbNnZs4x+Hnp8mY7EzLcKxx5pc0aPIrz2+bNIoYp9Y6bphHCDvXKjjHCIn+1YOjSeY6Hjy5UYr+i+F8FOD4M2VF8L61ZB8GFltJX2s05fYLpRnJjOJIyeErD7zT4ELEop02nqiVpSWjNkdQ2q0a+0nBuvaBLH2tuq3e9A482Ox93IwR05rXm8W6stFznt1fCYamB+69p/Md4PMFWZsGvx36rTVQ/g8GopSejgPbb5jj5FSjatpQamsbq+kjBu9viywAcaiEcXMPe4cx5juVxdSsuj3iC7y+L8lFRkPCyfdbH3X8L/ooBERUxfBERAEREAREQBERAEREAREQBERAEREBndBacqdWatoLDSu3DUyfaSdI4wMvefg0ErcWnpqSgo6e326IQ0VJG2GnjHRjeA8zzJ6klU56LFgENsu2qpo/bmcKGkcRyHB0pH8o+auV8sNPHJUVMgjghY6SVx5BrRkn5BZfbWRKy2NEfTMrtvIlbbHHj05+bKk9JLWDrZaYtK0ExbU17O0rHNPFsOeDP4iMnwHitdlmNaX2o1Lqm4XupJ3qqYua38DOTWj4DAWHV/iY6x6lWjQ4eNHGpVa9MIisvYXe9FadvFdLregdIySICBzoDJu88gDBxnhxx0XVFavQnk9FqQCz3OvtFwjr7bUyU1RGctew/Q948Crjsl50/tQs7bLfoRS3aBhdFJFwLT1dH3jqWHu4LE3W/wCyGorZpG6cqcOcSHQxljTx543xj5Lss+sNmlhkdXWex1ba0NIY5zSTxHIEvIGe9dlP6b0lJOL5r11OG/Wxb0YNSXJ8PWniQLWuk7rpWvEFcwSU8hPYVUfGOYeB6HvB4hZ7ZrtFqtOD9FXWP9I2OUFklPIN4xg890HgR3t/IqX6U2gWfV0M+n9W0dJCKl/2Q92J/cM/ceOjuqhe0fZ3cdLyvraQSVtoc7DZg324f2ZB0PjyP0Xmde7+pU9Y/VeZ7hbv/pXrSX0fkbB2p0Nf2Bt25HQywRmLs/d7LdG7jPHkpNFG2NgY0YAGAons0pjS2K1wHOY7dDnPTLQcfVSyR7IxmR7WDxKyPtPLey0kv9V9jE5K/WnFPqzsBHEOa17XAtc1wyHA8wR1BC1c246E/U/UDaugZ/sa4l0lLjj2TvvRH4Z4d4IWzUdTFKcRbzvEN4LG6505T6t0nXWCYNbJK3fpJHf2c7fcPwPFp8CVwbMzHjXbkvhZ37KzXj2qMvhZpoi7qymno6uakqY3RTwyOjkY7m1wOCD5rpWyNoZHTV0lst/orpDxdTTNeR+IdR5jIWz7JWiSOopnZY4Nlid3tIBB+RWp62Q2eVnr+gbNVF285sBp3nxjcW/luq82HbpZKt8mjPe0NOtUbVzTKp2z6bZZNTCupIRHb7mDPE1o9mN+ftGD4HiPBwUFWxe060fpvQNwhawOnoR67Aeo3R9oB8W5P8IWui4do43u97iuXNFjszK95x4zfNcGERFwlgEREAREQBERAEREAREQBERAEReyyUprrzQ0QGTUVEcWP3nAf1QG3mzm1Cx7OrBbN3ce2jbPKCP7SX2z/wBQHksFt7u7rRsxrRE8smuErKNmDg7py5/8rceantSA2Z7GjDWHcaO4DgPyVIelZW7tHp22A+8Zqlwz+6wfk5ZDCXvG0HN+ZjsH9xtFzfi2VRs500NWarprK6p9WZIHPfIBkgNGeHipbtt2XR6BbTVFJWyVMMkvYyB5Dix27vDiOhGVXFurau3VsVbQ1ElPUxO3o5I3Yc0q6tD6totoNqqNK6ra01szMiRoAM27ye3ulbzwODhlbamEbNY9ehqL5zramuMev5Ko0FJSw60s76yETQeuRh7CMg5cBy81a/pWXG11NTY6WjoxBPHG9zjzO7kN59xIPDwUAk0tcNMbSrTbKtpfG+uhfTTgYbPH2gw4f1HQrK+kLLv62gZz3KJuT4l7z/kvShpTPXmml9/weJT3r4aPg039vyVuvZWWq50dNFVVdvqoIJf/AE5JInNa74Erz00nY1Ec26H7jw7dPI4OcLZK6Xu3bUNl89BaaOCCsjh3RFye2ZpDmtPTBAIafJR119pqlzJrrey0bXA1pVx7ENb3asulNpSuBq45I3Np53+06NoaTuvz7zMDryVPSsfFI6ORjmPYS1zXDBBHMFXH6O1heyKv1NKwhz2mhovEnBkd5DA/iK94spK1aPTx8upDnuuNEpT6ffoWzS1dUZJOxYDLJzIHIdwHRZSltpyJayR0rzx3TyC9NvpI6SIBo9o+8V6vNYja+f71lTnDlqYPf1XA+WNaxoDQAPBfQznhzXy57Ad0OBPcuQVT8Tya6+k1pttt1dBf6ZgbT3iMukA+7OzAf8wWu8yqlW1e3q0tu+y24PLQZrbIysiPUDO68f4XZ8lqotzs2/t8eMnzXA3ezMjt8aMnzXAK8dhFWZ9FVdIf/a12R8HsH9WlUcrk9Hsk2a+joJ4D9Hq/2S9MuHroR7YjvYU/XUs+jcxtS3tGh0bsse08i0jBHyJWr+q7YbNqa5Wo8qSqkiB7wHEA/LC2bHNUZt4p2wbSa2RgwKmGCc/F0Tc/UFWe36+EZ/8ACo9nLGnOv/pBERFmzVBERAEREAREQBERAEREAREQBSXZXE2baXpqNwJBudPnHhICo0pHswnZTbRtOTvOGtucGT4doF8lyZ5lyZuJUHM8h73Fa/elgT+tljj6C1A/OWRbB1AAmkGOTj+a179K9p/WyxydDagPlLJ/mslsP/Jl5MyWwv8AKl5MppdlNPNTVEdRTyuimjcHMe04LSORC60WuNebC6C1RadcWunZeKZkt2tkjagRh26Q8f2rPA8N5vf5KuNvLnO1+/eBAFJDu+I3c8PMlQyz3GstNyguFBM6GohdvMcPyPeD3K6qqitO1fSMdVSmKkvlI3cbk43X8+zf+w453XdD5qy7X3mlw0764+aS+6Kp1LEvVmvcfDybf2f0KKUh0Dqao0tqCKvjDn07/s6qIHHaRk8fMcwe8LC19JU0FbNRVkD4KiB5ZLG8Yc1w5groVfGTi9VzLOUVOLi+TLo2k6BqNU3ahv2koW1TbiGesdmQG8RwnPcCPe7iPFWzo+0UtrttHQUoApLfF2UZ/G7m558SclV5sLdcW7P3uq5HiA1RbQ8ePZ4+0/h3seeVYUL56traWmBZEOeF52vfGnG3ocJWfRdf5MltS6x/tm+Eev2/gytVc4mns6YdtKTwA5LiOnq6nDqmUxtP3GLvt1DFSR+zxeRxcvSvz2U4w4QKZyS4RPmOGOIANaOHXqvtChUbfieTwakpBX6YvNA5uRUW6ePHiYzj6gLSlbxTOAppyeQhkz/hK0ePMrUbAk3VJfP+jV+z8m6Zr5nCu3YFTmPSFyqiMCauaweO4zP/AHKklsTsso/UdndqjIw+ffqXeO+7A+jQtlseDllxa6HTtue7hyXjoiSZ4KmfSFaBrilIGN62QE/zD+iubwVLekHJva9jjJ4xW6nafNpd/wByttuv9GPn+Sl9nf8A7y8v7RXaIiypsAiIgCIiAIiIAiIgCIiAIiIAu+31DqSvp6tnvQytkHxBB/ouhEBvL28dUGVcJ3oqhjZmHvDgHD81SnpXURfR6cugHBnb0rj5tePzcp5sXu4vezKzzudvTUjHUU3gYz7Of4S1ePb/AGl922W1r4mb8ttnZWNAHHd4sf8AR2fJZPD/AG20HB+LRkML9ttFwfLVo1VRFP7dsi1nXWSnvEdNSx01QA5nazhhwRkZzw5dMrXRhKT0itTWynGC1k9CALMaR1BXaavMdyoXA49mWJ3uysPNp/z6HippRbGtQSxOfU3S1U5HJvaPef5W4ChOq9PXLTV2fbbnEGyABzHsOWSNPJzT1CkdVtWkmmiKN1N2sIyTLb13p6h2h6ag1VpsB1yjh9tn3qhrRxjd/etHL8Q8lSAY8yCMNO+Tjdxxz3KUbN9YVOk7vvnfkt85AqYQePDk9vc4f+Fblfs7tuq9U2nVFkqIBDNPHPWRt4NnZvAmVnQHh7Te/wA1NOKyF2kF3uq/v8nPXJ40uzm+70f9P+iV2W0eo263WSnbhtHSxxO/exl583EqW0dNHTQiOMfE968llG+6pmOC58pOfNe+aRkTDJI4NA55WP8AaTIlZmulco6L+EYm2122Nvq2fXwXxJLFEN6R7WjvJWIq7tJJIIqJuSeGccV90lrfI7tq55c78OVRvH3VrNnlV6LVntbXslduU8bpT3gcF6o849vAPguImtjaGsaAAvrkopadDy2uhjtUVTaHSt5rXnAgt878+O4cfVaVraP0g77HZ9m09C1wFVd5W08Y69m0h0h+jR/EtXFrdiVOGPvPqzYbDqcMfefVnqtVFPcrnTW+maXTVMrYmAd7jhbRsghpIYaKnx2NNEyCPxawBufoqa2D2V1TqGa+ys+wt7CIyR70zwQAPgMn5K5ieS32wqNIyufkVntFk6yjSunFnLWlzmtHEk4Wv+2Ss9c2k3hwdvNhlFO3/ltDPzBWwLamGghqLlU4EFFA+pkJ7mNJA8zgea1Wr6mWtrp6yc5lnkdK897nHJ/NRbes70K/+kvs5S1Gdj8joREWeNOEREAREQBERAEREAREQBERAEREBdnos37srlddMzP9iriFVTA9JI/eA+LD/Kr4dBDWRy0NUA6CqjdBKD1a8Fp/NaXaVvNVp7UVDeqI/bUkzZADycOrT4EZHmtyLTcqG82mkvFtk36SrjEkRzxbnm0+IOQfgsztqmVdkciPrQy+26HXbHIj6aNMr9baiz3uutVU0tno53wvB72uI/op2Ns+r2aQh05C6mijia1rZw0l/AYBwTu5x1wpL6Tmk3MrYNZ0bMxVIbBX4+7KBhr/AIOaMfFviqRWixsjtK1ZB8y/pnXk1RnpqmZ2XWOqpZWySaguRc05biocAPIcFbFnqbbtR0UbdcZIYbtTDhKRxhk6PH7DuRHQ+Sope+w3auslziuNumMU8Z+IcOrXDqD3Lspv3G1Pinz9eJ8vx99Jw4SXJ+ujOu8W2stFynt1wgdDUwPLXsP5jvB5gqf7ENVy2+7s09V1BbRVjiICXY7KY8sHoHcj44UprILNtX0u2op+ypL9SMDWkni0/wDDeerD913Tl3ql7hR11nuklJWQSUtZTSYexww5rh/+5r3pLGsjZB6ro/X1PCcMuuVVi0fJrw+f4ZuBaKllJRzducOY73TzysbUVFRXT4y45PstHJeShrW3e30VxgO8K2CObh+JzRn65UotFvbSwiSQAyuHyWY2+oU5cresuP8AKMXKPYuUXzTPq10MdJCCRmUj2ivbklcdFzxWVlNzerOdy3jjkvpoyeYHieniuACTgDJVJ7b9qUApKjS+mKrtHygx11ZGfZDesbCOeeRPkF14WFPKnux5dTqwcKeXZurl1ZANt+rG6q1vPJSyF1uoR6tScchzQfaf/E7J+GFB4o3yysiiYXve4Na0DJJPIBfKt7Yvo8wNi1XdIiHnjbonDr/xj4D7vjx6Lf4mK7JRqrRtbra8OjefJE70jZGac01R2gNAmY3tKkj70zuLvlwb5LKLnieJ45X3DGZJWsBAB5k8gOpPgt5VXGipQXJH57fZLJtc3zbIXtruwtmgvUGkipus4jH/AMLMOf8AN24PmqCUv2salZqTVkslI8m3Ug9Xox3sB4v/AIjk/JRBYrNyPeL5T6H6BgY3u+PGHXr5hERch2BERAEREAREQBERAEREAREQBERAFbXo+6+bY7h+rN3nDbXWyZgkeeFPMeGfBruAPccHvVSoorqY3QcJ8mRXUxug4T5M3eulBRXW2VdqukHb0VVGYpmdcHqD0IOCD4LUfaRo+4aL1JLa6wOkgd9pSVOMNni6OHj0I6FW3sN2msrooNLajqS2raBHQ1Uh4SjkI3k/e6A9eR6ZszXWlbZrOwPsl2HYyNJfS1OMvppO/wAWnkR/UKgxrZ7Nu7G34XyZnsW2ezLuxt+B8maZIs7rfSl40ffJLVeacxyD2opW8Y5mZ4PYeoWCWjTTWqNKmmtUZHT15uFhukdxtsximZwI5te3q1w6gq4a+ls21fTLayjMVHfqRgaC84x/dvPVh+67pyVGrJabvVfYLtFcrfLuSsOHNPuyN6tcOoK6Kbd3uy4xfrVfM576HPvw4SXL8P5F57Eayogsk1muMLoa6z1RjkjkGHCN+SPIHPzCuAHLQehVX6MvVk1Ri/0wMNayn9Xq4s5cBza134gCPZd5KwrLUesUQB95nAqq9pcFyxYZEeKjw1+XT8GO2otb29NG+a+Z7V4L9erVYLc64Xivho6cDgXn2nnua3m4+AUX2qbRbfoinFLExlZepWh0dMT7MQPJ8mOncOZ8FrLqS/XbUVzfcbxWyVVQ/kXHg0dzRyA8AqDA2Q7krLOEfA69n7HlelZbwj9WWFtR2v1+oYprTYWSW+1P9mSQnE1QPEj3W+A8z0VVLuo6aorKllNSQSTzSHDI42lznHwAVtaD2XtpJY7jqljZJG4cygByAf7wjn+6PM9Fr8LActKqI+vmaKdmPgVceC+5h9l2z83Tsr5fYnMtoO9BAeDqoj8mePXkO9XM92+c4DQAAGgYDQOQA7ly9xecnHLAAGAB0A7gvnittgYMMSPDi3zMZtDaE8yer4RXJHOePAKC7Y9Wsslpk07QS5ulbGBVPYf93hP3P3nfRvxWW2g6xp9HUAbHuTXqdmaeA8RC0/2jx+TevwWvdZU1FZVy1dVM+aeZ5fJI85LnHmSqra+0U06K/wDv4LfY2zHqsi1eS/s6URFnDUhERAEREAREQBERAEREAREQBERAEREAREQHIJBBBwRyKvDZNtiEccVj1jMSxoDKe5EEub0DZe8ftc+/vVHIoMjHryIbli1RBkY1eRDcsWqN0NUWCy6y08LbeI21NI9vaUtVC4F8JI9+N3d3jkVrDtI2dX7RVWXVUXrdse/EFfCMxv7g78DvA+WVxoDaNqPR0gio5xVW8nL6Koy6M/u9WHxH1V9aO2qaP1VSOt9XLHbqidu7LRXHDoZc9A8+yfPBVZXDJwOC78PqirrrycB6RW/X9UaqIti9bbCrXcXGu0vV/oqSQbwp58vp3H9h4yWj45Cp/VGzvWOnHu/SNjqjADwqIG9rE4d+83IHnhWVOVVd8L4+HUsqMuq74Hx8OpiNL32v07d4rlb5MPbwew+7I3q1w6gq9aXatp6j07UXekmHrhiIjt8mS8SkcM97QeOe4LXYgg4PArL6Rr7Tbb1HU3q0i6UYaQ6EvLcHo7xx3HguzfbrlU/hlzR5ycKnIalNatetD5LL5qm+TTshqrncKqQySdmwvc4n4cgptp3ZFd6hzZtQVUNrg6xNcJJ3eG6ODfM+SsjTWrtLV9GILLW0NvZjjS7jaZw8uAd8QSs4xrnt32DfB5FpyD8lc4Wy6LEpTs1+SKfO2vkVNwhW182YvTmn7LpyB0VmohC5w3ZKh535pPi7oPAYCyY4lcS4ibvSuZE3ve4NH1UavmvNK2hjhLcRWTjlDRgPJPi73R81e9pi4kdE0kZ/s8vNnq05Mk7I3SPDI2uc4ngAFDdfbQrfpmN9Fa3QXC8kYJ96GlPeTyc/w5Dr3KvdX7Tb3eYn0dvAtVC8YcyF32kg/afz8hgKCKjzdsysThTwXj1NDs/YUamp38X4dP8A09Fyrau5V01dX1ElTUzOL5JZHZc4rzoioTRBERAEREAREQBERAEREAREQBERAEREAREQBERAEREAREQGe01rLVGnHA2a91lKwHPZCTejPxYct+isWxekFqajLRcbVbq4DgXR70Dz/hJH0VOIorKK7OMopkVlFdnGcUy/ptr+zK/De1Rs67WY85GMie4/xDccsXcL16O9Wd5ultS0ruvq8uM+RkIVKovsK1DkI0xjy1/lkn1rNoSQsGj6O+w+17brhNG4EeAaP6qOxVNREMRTys/deQupFISnZLPNMcyzSSfvOJXWiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgCIiAIiIAiIgP/2Q=="

def _speak_bg(text, lang, result):
    try:    result.append(speak(text, lang))
    except: result.append("")

# ── PIL for page icon ─────────────────────────────────────────
_logo_pil = None
try:
    import io
    from PIL import Image as _PILImage
    _logo_pil = _PILImage.open(io.BytesIO(base64.b64decode(_LOGO_B64)))
except Exception:
    pass

st.set_page_config(
    page_title="Samridhi AI – M-CADWM",
    page_icon=_logo_pil if _logo_pil else "🏛️",
    layout="wide",
)

load_dotenv()
_groq_key = os.getenv("GROQ_API_KEY")
if not _groq_key:
    try:    _groq_key = st.secrets["GROQ_API_KEY"]
    except: _groq_key = None
if not _groq_key:
    st.error("GROQ_API_KEY not set."); st.stop()

try:    vector_db = get_vector_db()
except Exception as e: st.error(f"FAISS error: {e}"); st.stop()

llm         = get_llm()
feedback_db = get_feedback_db()
web_cache   = get_web_cache()
analytics   = get_analytics()
expansions  = get_expansions()
pipeline    = Pipeline(llm, vector_db, feedback_db, web_cache, analytics,
                       expansion_store=expansions)

def _init():
    for k,v in {
        "lang":"en","messages":[],"pending_feedback":{},
        "reingest_done":set(),"last_answer":"",
        "rate_bucket":RateLimiter.make_bucket(),
        "tts_enabled":False,"followup_queue":None,
    }.items():
        if k not in st.session_state: st.session_state[k]=v
_init()

lang = st.session_state.lang
ui   = UI[lang]

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
#MainMenu,footer,header,[data-testid="stToolbar"],
[data-testid="stDecoration"],[data-testid="stStatusWidget"]{{
  display:none!important;visibility:hidden!important;
}}
:root{{
  --bg:#0E1117;--surface:#1A1D2E;--surface2:#12141F;
  --border:rgba(255,255,255,0.08);--text:#E8ECF4;--muted:#8892A4;
  --accent:#4A90D9;--accent-hover:#5BA3EC;
  --green:#4ade80;--amber:#fbbf24;--radius:12px;
  --font:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
}}
*,*::before,*::after{{box-sizing:border-box;}}
html,body,.stApp,[data-testid="stAppViewContainer"]{{
  background:var(--bg)!important;color:var(--text)!important;
  font-family:var(--font)!important;font-size:15px!important;line-height:1.6!important;
}}
.block-container{{padding:0!important;max-width:100%!important;}}
[data-testid="stMain"]{{background:var(--bg)!important;padding:0!important;}}

/* Hide functional lang buttons — HTML ones are shown instead */
[data-testid="stSidebar"] [data-testid="column"] .stButton>button {{
  display: none !important;
}}

/* Sidebar */
[data-testid="stSidebar"]{{
  background:var(--surface2)!important;
  border-right:1px solid var(--border)!important;
  min-width:260px!important;max-width:260px!important;
}}
[data-testid="stSidebar"]>div{{
  background:var(--surface2)!important;
  padding:20px 14px!important;
}}

/* Brand */
.sidebar-brand{{
  display:flex;flex-direction:column;align-items:center;
  padding-bottom:16px;border-bottom:1px solid var(--border);margin-bottom:8px;
}}
.sidebar-brand img{{
  width:64px;height:64px;object-fit:contain;
  border-radius:8px;margin-bottom:8px;
}}
.sidebar-brand .name{{font-weight:700;font-size:16px;color:var(--text);}}
.sidebar-brand .sub{{font-size:11px;color:var(--muted);margin-top:2px;}}

/* Language buttons */
.lang-toggle{{display:flex;gap:6px;margin-bottom:4px;width:100%;}}
.lang-btn{{
  flex:1;padding:7px 4px;border-radius:7px;
  border:1px solid var(--border);background:transparent;
  color:var(--muted);font-size:12px;font-weight:600;
  cursor:pointer;text-align:center;transition:all 0.15s;
}}
.lang-btn.active{{background:var(--accent);border-color:var(--accent);color:#fff;}}

/* Divider */
.divider{{height:1px;background:var(--border);margin:10px 0;}}

/* Voice toggle */
[data-testid="stSidebar"] [data-testid="stToggle"] label{{
  color:var(--muted)!important;font-size:13px!important;
}}

/* Expander / accordion */
[data-testid="stExpander"]{{border:none!important;background:transparent!important;box-shadow:none!important;}}
[data-testid="stExpander"] summary{{
  color:var(--muted)!important;font-size:13px!important;
  padding:8px 4px!important;border-radius:6px!important;background:transparent!important;
}}
[data-testid="stExpander"] summary:hover{{background:rgba(255,255,255,0.04)!important;}}

/* Sidebar buttons (starters) */
[data-testid="stSidebar"] .stButton>button{{
  background:transparent!important;border:1px solid var(--border)!important;
  color:var(--text)!important;border-radius:6px!important;
  font-size:12px!important;text-align:left!important;
  width:100%!important;padding:6px 8px!important;
  line-height:1.4!important;transition:all 0.15s!important;
}}
[data-testid="stSidebar"] .stButton>button:hover{{
  background:rgba(74,144,217,0.1)!important;border-color:var(--accent)!important;
}}

/* New conversation — primary blue — override above */
[data-testid="stSidebar"] .new-conv-btn button,
[data-testid="stSidebar"] .new-conv-btn .stButton>button{{
  background:var(--accent)!important;border-color:var(--accent)!important;
  color:#fff!important;font-weight:600!important;font-size:13px!important;
  text-align:center!important;border-radius:8px!important;
  padding:8px 12px!important;width:100%!important;
}}
[data-testid="stSidebar"] .new-conv-btn button:hover,
[data-testid="stSidebar"] .new-conv-btn .stButton>button:hover{{
  background:var(--accent-hover)!important;
}}

/* Chat header */
.chat-header{{
  padding:16px 24px 12px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:10px;
  background:var(--bg);flex-shrink:0;
}}
.chat-header img{{width:28px;height:28px;object-fit:contain;border-radius:6px;}}
.chat-header .title{{font-size:16px;font-weight:700;color:var(--text);}}
.chat-header .subtitle{{font-size:12px;color:var(--muted);}}

/* Chat messages */
[data-testid="stChatMessage"]{{
  background:var(--surface)!important;border:1px solid var(--border)!important;
  border-radius:var(--radius)!important;border-bottom-left-radius:4px!important;
  padding:12px 16px!important;font-size:14px!important;line-height:1.65!important;
  animation:fadeUp 0.2s ease;
}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(6px);}}to{{opacity:1;transform:translateY(0);}}}}

/* Source badge */
.src-badge{{display:inline-block;font-size:11px;font-weight:600;padding:2px 9px;border-radius:20px;margin-bottom:8px;}}
.src-green{{background:rgba(74,222,128,0.12);color:var(--green);}}
.src-amber{{background:rgba(251,191,36,0.12);color:var(--amber);}}
.src-grey{{background:rgba(148,163,184,0.10);color:var(--muted);}}

/* Actions */
.msg-actions{{display:flex;gap:6px;margin-top:10px;flex-wrap:wrap;align-items:center;}}
.action-btn{{
  background:none;border:1px solid var(--border);border-radius:5px;
  padding:3px 10px;font-size:11px;color:var(--muted);cursor:pointer;transition:all 0.15s;
}}
.action-btn:hover{{background:var(--accent);color:#fff;border-color:var(--accent);}}

/* Follow-ups */
.follow-ups{{margin-top:10px;display:flex;flex-direction:column;gap:5px;}}
.follow-label{{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;}}
.follow-btn{{
  background:rgba(74,144,217,0.08);border:1px solid rgba(74,144,217,0.20);
  border-radius:7px;padding:7px 12px;font-size:13px;color:#93c5fd;
  cursor:pointer;text-align:left;transition:all 0.15s;line-height:1.4;width:100%;
}}
.follow-btn:hover{{background:var(--accent);color:#fff;border-color:var(--accent);}}

/* Chat input */
[data-testid="stChatInput"]{{
  background:var(--surface2)!important;
  border-top:1px solid var(--border)!important;
}}
[data-testid="stChatInputTextArea"]{{
  background:var(--surface)!important;border:1px solid var(--border)!important;
  border-radius:12px!important;color:var(--text)!important;font-size:14px!important;
}}
[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"]{{
  background:var(--accent)!important;border-radius:8px!important;
}}

/* Footer */
.input-hint{{text-align:center;font-size:11px;color:var(--muted);padding:6px 24px;}}

/* Scrollbar */
::-webkit-scrollbar{{width:4px;}}
::-webkit-scrollbar-track{{background:transparent;}}
::-webkit-scrollbar-thumb{{background:rgba(255,255,255,0.1);border-radius:4px;}}

/* About */
.about-text{{font-size:11px;color:var(--muted);line-height:1.7;padding:4px;}}
.about-text a{{color:var(--accent);text-decoration:none;}}
</style>
""", unsafe_allow_html=True)

components.html(BRIDGE_JS, height=0)

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    # Brand with logo
    st.markdown(f"""
    <div class="sidebar-brand">
      <img src="data:image/png;base64,{_LOGO_B64}" alt="Samridhi AI">
      <div class="name">Samridhi AI</div>
      <div class="sub">M-CADWM &amp; SMIS</div>
    </div>""", unsafe_allow_html=True)

    # Language toggle — HTML visual + Streamlit functional buttons hidden
    st.markdown(f"""
    <div class="lang-toggle">
      <button class="lang-btn {'active' if lang=='en' else ''}" onclick="document.querySelector('[data-testid=stSidebar] [data-testid=column]:first-child button').click()">🇬🇧 EN</button>
      <button class="lang-btn {'active' if lang=='hi' else ''}" onclick="document.querySelector('[data-testid=stSidebar] [data-testid=column]:last-child button').click()">🇮🇳 HI</button>
    </div>""", unsafe_allow_html=True)

    # Hidden functional buttons that actually trigger rerun
    _c1, _c2 = st.columns(2)
    with _c1:
        if st.button("🇬🇧 EN", key="btn_en",
                     type="primary" if lang=="en" else "secondary",
                     use_container_width=True):
            st.session_state.lang="en"; st.session_state.messages=[]; st.rerun()
    with _c2:
        if st.button("🇮🇳 HI", key="btn_hi",
                     type="primary" if lang=="hi" else "secondary",
                     use_container_width=True):
            st.session_state.lang="hi"; st.session_state.messages=[]; st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Voice toggle
    st.session_state.tts_enabled = st.toggle("Voice responses", value=st.session_state.tts_enabled)

    # Common queries
    _starters = (cfg["ui"]["starter_questions_hi"] if lang=="hi"
                 else cfg["ui"]["starter_questions_en"])
    with st.expander("❓ Common queries", expanded=False):
        for _sq in _starters:
            if st.button(_sq, key=f"sq_{hash(_sq)}", use_container_width=True):
                st.session_state.followup_queue=_sq; st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # New conversation
    st.markdown('<div class="new-conv-btn">', unsafe_allow_html=True)
    if st.button("+ New conversation", use_container_width=True, key="new_conv"):
        st.session_state.messages=[]
        st.session_state.pending_feedback={}
        st.session_state.last_answer=""
        st.session_state.followup_queue=None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # About
    with st.expander("About", expanded=False):
        st.markdown("""<div class="about-text">
          <strong style="color:var(--text)">Samridhi AI</strong> &nbsp;v2.0<br>
          AI Assistant for M-CADWM &amp; SMIS<br>
          CADWM Wing, Department of Water Resources,<br>
          River Development &amp; Ganga Rejuvenation<br>
          Ministry of Jal Shakti, Government of India<br>
          <a href="https://cadwm.gov.in" target="_blank">cadwm.gov.in</a><br>
          <span style="font-size:10px;opacity:0.6;">&copy; 2025 All Rights Reserved</span>
        </div>""", unsafe_allow_html=True)

    # PDF export
    if _HAS_PDF and st.session_state.last_answer:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        if st.button("⬇ Export PDF", use_container_width=True):
            try:
                import io, datetime as _dt
                buf=io.BytesIO()
                doc=SimpleDocTemplate(buf,pagesize=A4)
                styles=getSampleStyleSheet()
                clean=re.sub(r"#{1,6}\s*","",st.session_state.last_answer)
                clean=re.sub(r"\*\*(.*?)\*\*",r"\1",clean)
                story=[Paragraph("Samridhi AI — M-CADWM & SMIS",styles["Title"]),Spacer(1,12),
                       Paragraph(f"Generated: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",styles["Normal"]),Spacer(1,12)]
                for para in clean.split("\n\n"):
                    para=para.strip()
                    if para:
                        story.append(Paragraph(para.replace("\n","<br/>"),styles["Normal"]))
                        story.append(Spacer(1,8))
                doc.build(story)
                st.download_button("⬇ Download",data=buf.getvalue(),
                    file_name=f"samridhi_{int(time.time())}.pdf",mime="application/pdf")
            except Exception as e:
                st.error(f"PDF error: {e}")

    # Operator panel
    if st.query_params.get("operator")=="1":
        _op_pw=cfg.get("operator",{}).get("password","samridhi-admin")
        if not st.session_state.get("op_unlocked"):
            _pwd=st.text_input("Password",type="password",key="op_pwd")
            if st.button("Unlock",key="op_btn"):
                if _pwd==_op_pw: st.session_state.op_unlocked=True; st.rerun()
                else: st.error("Incorrect password.")
        else:
            st.markdown("**System Status**")
            fb_s=feedback_db.stats();wc_s=web_cache.stats()
            ex_s=expansions.stats();ana_r=analytics.recent(5)
            faiss_ok=(BASE_DIR/"faiss_index").exists()
            st.markdown(
                f"- FAISS: {'✅' if faiss_ok else '❌'}\n"
                f"- Feedback: {fb_s['total']} ({fb_s['positive']} positive)\n"
                f"- Web cache: {wc_s['total_entries']}\n"
                f"- Expansions: {ex_s['enabled']}/{ex_s['total']}")
            if ana_r:
                st.markdown("**Recent queries**")
                for rec in reversed(ana_r):
                    st.caption(f"`{rec.get('layer','?')}` {rec.get('confidence',0):.2f} "
                               f"{rec.get('response_ms',0):.0f}ms — {rec.get('query','')[:40]}")
            if st.button("Lock",key="op_lock"):
                st.session_state.op_unlocked=False; st.rerun()

# ══════════════════════════════════════════════════════════════
# CHAT HEADER
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="chat-header">
  <img src="data:image/png;base64,{_LOGO_B64}" alt="">
  <div>
    <div class="title">Samridhi AI</div>
    <div class="subtitle">AI Assistant — M-CADWM &amp; SMIS</div>
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
_SRC_MARKERS=[
    UI["en"]["src_faiss"],UI["en"]["src_live"],UI["en"]["src_general"],
    UI["hi"]["src_faiss"],UI["hi"]["src_live"],UI["hi"]["src_general"],
]

def _strip_source(text):
    for m in _SRC_MARKERS:
        if m and text.rstrip().endswith(m.strip()):
            return text[:len(text.rstrip())-len(m.strip())].rstrip()
    return text

def _source_badge(layer):
    badge_map={
        "faiss":("src-green",ui.get("badge_faiss","✦ M-CADWM Official Documents")),
        "live":("src-amber",ui.get("badge_live","◉ cadwm.gov.in (live)")),
        "fallback":("src-grey",ui.get("badge_general","◈ General Knowledge")),
        "cache":("src-grey",ui.get("badge_cache","◇ Cached")),
    }
    if layer not in badge_map: return
    cls,label=badge_map[layer]
    st.markdown(f'<span class="src-badge {cls}">{label}</span>',unsafe_allow_html=True)

def _follow_ups(fups,msg_idx):
    if not fups: return
    for _fi,fq in enumerate(fups):
        if st.button(fq,key=f"fup_{msg_idx}_{_fi}",use_container_width=True):
            st.session_state.followup_queue=fq; st.rerun()

def _copy_btn(content):
    b64=base64.b64encode(content.encode()).decode()
    st.markdown(
        f'<div class="msg-actions">'
        f'<button class="action-btn" data-copy="{b64}" '
        f'onclick="var t=atob(this.getAttribute(\'data-copy\'));'
        f'navigator.clipboard.writeText(t).then(()=>{{this.textContent=\'✓ Copied\';}});">'
        f'{ui.get("copy_label","Copy answer")}</button>'
        f'</div>',
        unsafe_allow_html=True,
    )

def _feedback(i,layer=""):
    if layer not in _FEEDBACK_LAYERS: return
    if i not in st.session_state.pending_feedback: return
    pf=st.session_state.pending_feedback[i]
    rk=f"rated_{i}"
    if rk not in st.session_state:
        c1,c2,_=st.columns([1,1,8])
        with c1:
            if st.button("👍",key=f"up_{i}"):
                feedback_db.record(pf["q"],pf["a"],"up",lang)
                st.session_state[rk]="up"; st.rerun()
        with c2:
            if st.button("👎",key=f"dn_{i}"):
                feedback_db.record(pf["q"],pf["a"],"down",lang)
                st.session_state[rk]="down"; st.rerun()
    else:
        st.caption(ui["fb_up"] if st.session_state[rk]=="up" else ui["fb_dn"])

# ══════════════════════════════════════════════════════════════
# FOLLOW-UP QUEUE
# ══════════════════════════════════════════════════════════════
def _process_followup():
    q=st.session_state.followup_queue
    if not q: return
    st.session_state.followup_queue=None
    st.session_state.messages.append({"role":"user","content":q,"follow_ups":[],"layer":""})
    r=pipeline.run(q,lang,st.session_state.messages,st.session_state.rate_bucket,ui)
    pipeline.maybe_reingest(lang,st.session_state.reingest_done)
    st.session_state.last_answer=r.answer
    _idx=len(st.session_state.messages)
    st.session_state.pending_feedback[_idx]={"q":q,"a":r.answer}
    if len(st.session_state.pending_feedback)>_PENDING_FEEDBACK_MAX:
        del st.session_state.pending_feedback[min(st.session_state.pending_feedback)]
    st.session_state.messages.append(
        {"role":"assistant","content":r.answer,"follow_ups":r.follow_ups,"layer":r.layer})
_process_followup()

# Welcome
if not st.session_state.messages:
    st.session_state.messages=[
        {"role":"assistant","content":ui["welcome"],"follow_ups":[],"layer":""}]

# ══════════════════════════════════════════════════════════════
# CHAT HISTORY
# ══════════════════════════════════════════════════════════════
for _i,_msg in enumerate(st.session_state.messages):
    with st.chat_message(_msg["role"]):
        _layer=_msg.get("layer","")
        _is_sub=_layer in _FEEDBACK_LAYERS
        if _msg["role"]=="assistant" and _is_sub:
            _source_badge(_layer)
        _display=_strip_source(_msg["content"]) if _msg["role"]=="assistant" else _msg["content"]
        st.markdown(_display)
        _follow_ups(_msg.get("follow_ups",[]),_i)
        if _msg["role"]=="assistant" and _is_sub:
            _copy_btn(_msg["content"])
        if _msg["role"]=="assistant":
            _feedback(_i,_layer)

# ══════════════════════════════════════════════════════════════
# USER INPUT
# ══════════════════════════════════════════════════════════════
if question:=st.chat_input(ui["placeholder"]):
    st.session_state.messages.append({"role":"user","content":question,"follow_ups":[],"layer":""})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        _tts_result:list=[]
        _tts_thread=None
        with st.spinner(ui["spinner"]):
            r=pipeline.run(question,lang,st.session_state.messages,st.session_state.rate_bucket,ui)
        if st.session_state.tts_enabled and r.layer in _FEEDBACK_LAYERS:
            _tts_thread=threading.Thread(target=_speak_bg,args=(r.answer,lang,_tts_result),daemon=True)
            _tts_thread.start()
        _source_badge(r.layer)
        st.markdown(_strip_source(r.answer))
        st.session_state.last_answer=r.answer
        _follow_ups(r.follow_ups,len(st.session_state.messages))
        if r.layer in _FEEDBACK_LAYERS:
            _copy_btn(r.answer)
        _idx=len(st.session_state.messages)
        st.session_state.pending_feedback[_idx]={"q":question,"a":r.answer}
        if len(st.session_state.pending_feedback)>_PENDING_FEEDBACK_MAX:
            del st.session_state.pending_feedback[min(st.session_state.pending_feedback)]
        _feedback(_idx,r.layer)
        st.session_state.messages.append(
            {"role":"assistant","content":r.answer,"follow_ups":r.follow_ups,"layer":r.layer})
        pipeline.maybe_reingest(lang,st.session_state.reingest_done)
        if _tts_thread:
            _tts_thread.join(timeout=15)
            if _tts_result: autoplay_audio(_tts_result[0],st)

# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="input-hint">
  Copyright &copy; 2025 &nbsp;&middot;&nbsp; All Rights Reserved &nbsp;&middot;&nbsp;
  CADWM Wing, Department of Water Resources, River Development &amp;
  Ganga Rejuvenation, Ministry of Jal Shakti, Government of India
</div>""", unsafe_allow_html=True)
