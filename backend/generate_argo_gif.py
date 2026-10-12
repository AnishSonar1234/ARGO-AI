import math
import os
import random
from PIL import Image, ImageDraw, ImageFont

WIDTH = 800
HEIGHT = 500
FRAMES = 80
FPS = 16
DURATION_MS = int(1000 / FPS)

# Color palette
BG_SPACE = (6, 11, 25)
OCEAN_SURF = (14, 116, 144)
OCEAN_MID = (7, 45, 85)
OCEAN_DEEP = (2, 8, 22)
FLOAT_YELLOW = (245, 158, 11)
FLOAT_DARK = (15, 23, 42)
CYAN_ACCENT = (6, 182, 212)
EMERALD_ACCENT = (16, 185, 129)
WHITE = (248, 250, 252)
MUTED = (148, 163, 184)
GRID_LINE = (24, 45, 75)

try:
    font_title = ImageFont.truetype("arialbd.ttf", 15)
    font_bold = ImageFont.truetype("arialbd.ttf", 12)
    font_md = ImageFont.truetype("arial.ttf", 13)
    font_sm = ImageFont.truetype("arial.ttf", 11)
    font_mono = ImageFont.truetype("consola.ttf", 11)
    font_mono_bold = ImageFont.truetype("consolab.ttf", 12)
except Exception:
    font_title = ImageFont.load_default()
    font_bold = ImageFont.load_default()
    font_md = ImageFont.load_default()
    font_sm = ImageFont.load_default()
    font_mono = ImageFont.load_default()
    font_mono_bold = ImageFont.load_default()

base_bg = Image.new("RGB", (WIDTH, HEIGHT))
draw_bg = ImageDraw.Draw(base_bg)

# Sky / Space: Y 0 to 142
for y in range(142):
    t = y / 142.0
    r = int(BG_SPACE[0] * (1 - t * 0.2))
    g = int(BG_SPACE[1] * (1 + t * 0.4))
    b = int(BG_SPACE[2] * (1 + t * 0.8))
    draw_bg.line([(0, y), (WIDTH, y)], fill=(r, g, b))

# Subtle stars in sky
random.seed(42)
for _ in range(35):
    sx = random.randint(10, WIDTH - 10)
    sy = random.randint(75, 135)
    draw_bg.ellipse([sx, sy, sx + 1, sy + 1], fill=(200, 225, 255))

# Ocean: Y 142 to HEIGHT
for y in range(142, HEIGHT):
    t = (y - 142) / float(HEIGHT - 142)
    if t < 0.35:
        st = t / 0.35
        r = int(OCEAN_SURF[0] * (1 - st) + OCEAN_MID[0] * st)
        g = int(OCEAN_SURF[1] * (1 - st) + OCEAN_MID[1] * st)
        b = int(OCEAN_SURF[2] * (1 - st) + OCEAN_MID[2] * st)
    else:
        st = (t - 0.35) / 0.65
        r = int(OCEAN_MID[0] * (1 - st) + OCEAN_DEEP[0] * st)
        g = int(OCEAN_MID[1] * (1 - st) + OCEAN_DEEP[1] * st)
        b = int(OCEAN_MID[2] * (1 - st) + OCEAN_DEEP[2] * st)
    draw_bg.line([(0, y), (WIDTH, y)], fill=(r, g, b))

def draw_float(draw, x, y, bladder_inflated=False, antenna_active=False):
    hw = 18
    hh = 52
    
    # Body
    draw.rounded_rectangle([x - hw//2, y - hh//2, x + hw//2, y + hh//2], radius=4, fill=FLOAT_YELLOW, outline=(255, 255, 255, 180), width=1)
    
    # Collars
    draw.rectangle([x - hw//2 - 2, y - hh//2 + 7, x + hw//2 + 2, y - hh//2 + 12], fill=FLOAT_DARK)
    draw.rectangle([x - hw//2 - 2, y + hh//2 - 13, x + hw//2 + 2, y + hh//2 - 8], fill=FLOAT_DARK)
    
    # Label "ARGO"
    draw.text((x - 13, y - 5), "ARGO", fill=FLOAT_DARK, font=font_mono_bold)
    
    # CTD Head on top
    draw.rectangle([x - 6, y - hh//2 - 8, x + 6, y - hh//2], fill=FLOAT_DARK, outline=(100, 116, 139), width=1)
    draw.ellipse([x - 3, y - hh//2 - 9, x + 3, y - hh//2 - 5], fill=CYAN_ACCENT)
    
    # Antenna Mast
    draw.line([(x, y - hh//2 - 8), (x, y - hh//2 - 28)], fill=(226, 232, 240), width=2)
    # Beacon
    tip_col = (239, 68, 68) if antenna_active else (148, 163, 184)
    draw.ellipse([x - 3, y - hh//2 - 31, x + 3, y - hh//2 - 25], fill=tip_col)
    
    # Variable Buoyancy Bladder
    br = 8 if bladder_inflated else 4
    bh = 12 if bladder_inflated else 5
    draw.ellipse([x - br, y + hh//2 + 1, x + br, y + hh//2 + 1 + bh], fill=(13, 148, 136), outline=(45, 212, 191), width=1)

def draw_satellite(draw, sx, sy, active=False):
    # Main bus
    draw.rectangle([sx - 11, sy - 7, sx + 11, sy + 7], fill=(226, 232, 240), outline=(56, 189, 248), width=1)
    draw.rectangle([sx - 7, sy - 4, sx + 7, sy + 4], fill=(51, 65, 85))
    
    # Solar Arrays
    draw.rectangle([sx - 34, sy - 5, sx - 14, sy + 5], fill=(30, 58, 138), outline=(96, 165, 250), width=1)
    draw.line([(sx - 24, sy - 5), (sx - 24, sy + 5)], fill=(96, 165, 250), width=1)
    
    draw.rectangle([sx + 14, sy - 5, sx + 34, sy + 5], fill=(30, 58, 138), outline=(96, 165, 250), width=1)
    draw.line([(sx + 24, sy - 5), (sx + 24, sy + 5)], fill=(96, 165, 250), width=1)
    
    # Comm Dish
    draw.arc([sx - 7, sy + 5, sx + 7, sy + 15], start=0, end=180, fill=WHITE, width=2)
    draw.line([(sx, sy + 7), (sx, sy + 13)], fill=WHITE, width=2)
    
    if active:
        draw.ellipse([sx - 3, sy + 13, sx + 3, sy + 19], fill=(250, 204, 21))

frames = []

depth_markers = [
    (144, "0m", "Sea Surface"),
    (215, "500m", "Epipelagic"),
    (290, "1,000m", "Parking & Drift"),
    (365, "1,500m", "Mesopelagic"),
    (440, "2,000m", "Deep CTD Profile")
]

waypoints = [
    (170, 150),  # deployment
    (260, 290),  # reached 1000m
    (440, 290),  # finished drift
    (510, 440),  # reached 2000m
    (590, 150),  # reached surface
    (640, 150)   # satellite uplink
]

for f in range(FRAMES):
    im = base_bg.copy()
    draw = ImageDraw.Draw(im)
    
    # 1. Top Control HUD Bar
    draw.rounded_rectangle([16, 10, WIDTH - 16, 64], radius=10, fill=(15, 23, 42, 240), outline=(56, 189, 248, 150), width=1)
    
    # 2. Depth Grid & Layer labels
    for dy, dval, dzone in depth_markers:
        draw.line([(140, dy), (WIDTH - 20, dy)], fill=GRID_LINE, width=1)
        # Offset label slightly to avoid colliding with waves
        label_y = dy - 16 if dy == 144 else dy - 7
        draw.text((16, label_y), f"{dval:>6}", fill=CYAN_ACCENT, font=font_mono_bold)
        draw.text((70, label_y), dzone, fill=MUTED, font=font_sm)

    # 3. Thermocline Temperature Curve (on left)
    temp_points = []
    for y_eval in range(144, 445, 8):
        t_depth = (y_eval - 144) / (440 - 144)
        temp = 2.0 + 24.0 * math.exp(-3.5 * t_depth)
        tx = 145 + int(temp * 1.5)
        temp_points.append((tx, y_eval))
    for i in range(len(temp_points) - 1):
        draw.line([temp_points[i], temp_points[i+1]], fill=(251, 113, 133), width=1)
    draw.text((188, 156), "Thermocline (26°C → 2.2°C)", fill=(251, 113, 133), font=font_mono)

    # 4. Ocean surface animated wave crests
    wave_pts = []
    for wx in range(0, WIDTH + 20, 10):
        wy = 144 + math.sin((wx * 0.035) + (f * 0.22)) * 3.0
        wave_pts.append((wx, wy))
    for i in range(len(wave_pts) - 1):
        draw.line([wave_pts[i], wave_pts[i+1]], fill=(125, 211, 252), width=2)

    # 5. Mission trajectory dashed ghost path
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i+1]
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = int(dist / 10)
        for s in range(steps):
            if s % 2 == 0:
                t1 = s / float(steps)
                t2 = min(1.0, (s + 0.6) / float(steps))
                x_a = p1[0] + (p2[0] - p1[0]) * t1
                y_a = p1[1] + (p2[1] - p1[1]) * t1
                x_b = p1[0] + (p2[0] - p1[0]) * t2
                y_b = p1[1] + (p2[1] - p1[1]) * t2
                draw.line([(x_a, y_a), (x_b, y_b)], fill=(56, 189, 248, 80), width=1)

    # 6. Satellite in Sky Orbit (Y = 104)
    sat_x = 640 + math.sin(f * 0.04) * 20
    sat_y = 104
    draw.text((sat_x - 42, sat_y - 20), "Iridium RUDICS", fill=(186, 230, 253), font=font_mono)

    # 7. Float State & Position based on Frame
    stage_pill = ""
    stage_title = ""
    stage_detail = ""
    bladder_inflated = False
    transmitting = False
    ctd_active = False

    if f < 20:
        prog = f / 19.0
        ep = 0.5 - 0.5 * math.cos(prog * math.pi)
        fx = 170 + ep * (260 - 170)
        fy = 150 + ep * (290 - 150)
        stage_pill = "STAGE 1"
        stage_title = "Descent to 1,000m Parking Depth"
        stage_detail = f"Buoyancy bladder deflates • Sinking rate ~10 cm/s • Current depth: {int(ep * 1000)}m"
        bladder_inflated = False
        
    elif f < 40:
        prog = (f - 20) / 19.0
        fx = 260 + prog * (440 - 260)
        fy = 290 + math.sin(prog * math.pi * 3) * 5
        stage_pill = "STAGE 2"
        stage_title = "Subsurface Drift (9 Days)"
        stage_detail = f"Neutrally buoyant at 1,000m • Mapping deep currents • Day {int(1 + prog*8)}/9"
        bladder_inflated = False
        
        # Subsurface current vectors
        for cx in [290, 360, 420]:
            cy = 310
            draw.line([(cx, cy), (cx + 26, cy)], fill=CYAN_ACCENT, width=1)
            draw.polygon([(cx + 26, cy), (cx + 21, cy - 3), (cx + 21, cy + 3)], fill=CYAN_ACCENT)
        draw.text((320, 320), "Deep Ocean Currents", fill=CYAN_ACCENT, font=font_mono)

    elif f < 52:
        prog = (f - 40) / 11.0
        ep = 0.5 - 0.5 * math.cos(prog * math.pi)
        fx = 440 + ep * (510 - 440)
        fy = 290 + ep * (440 - 290)
        stage_pill = "STAGE 3A"
        stage_title = "Descent to 2,000m Depth"
        stage_detail = f"Diving to abyssal profiling baseline • Current depth: {int(1000 + ep * 1000)}m"
        bladder_inflated = False

    elif f < 68:
        prog = (f - 52) / 15.0
        ep = 0.5 - 0.5 * math.cos(prog * math.pi)
        fx = 510 + ep * (590 - 510)
        fy = 440 - ep * (440 - 150)
        stage_pill = "STAGE 3B"
        stage_title = "CTD Profiling Ascent"
        cur_depth = int(2000 - ep * 2000)
        cur_temp = round(2.0 + 24.0 * math.exp(-3.5 * (cur_depth / 2000.0)), 1)
        stage_detail = f"Ascending • CTD Sampling: {cur_depth}m • Temp: {cur_temp}°C • Salinity: 35.1 PSU"
        bladder_inflated = True
        ctd_active = True
        
        ring_r = ((f - 52) % 4) * 8 + 6
        draw.ellipse([fx - ring_r, fy - ring_r, fx + ring_r, fy + ring_r], outline=EMERALD_ACCENT, width=2)
        draw.text((fx + 22, fy - 8), f"Depth: {cur_depth}m", fill=EMERALD_ACCENT, font=font_mono_bold)

    else:
        prog = (f - 68) / 11.0
        fx = 590 + prog * (640 - 590)
        fy = 150 + math.sin(prog * math.pi * 2) * 2.5
        stage_pill = "STAGE 4"
        stage_title = "Satellite Telemetry Uplink"
        stage_detail = "At sea surface • GPS fix acquired • Transmitting data packets to satellite"
        bladder_inflated = True
        transmitting = True
        
        for ring_i in range(3):
            wt = ((f + ring_i * 3) % 9) / 9.0
            wx = fx + (sat_x - fx) * wt
            wy = fy - 28 + (sat_y + 13 - (fy - 28)) * wt
            wr = 4 + wt * 12
            draw.arc([wx - wr, wy - wr, wx + wr, wy + wr], start=210, end=330, fill=CYAN_ACCENT, width=2)
        draw.line([(fx, fy - 28), (sat_x, sat_y + 13)], fill=(56, 189, 248, 120), width=1)

    # Draw Satellite & Float
    draw_satellite(draw, sat_x, sat_y, active=transmitting)
    draw_float(draw, int(fx), int(fy), bladder_inflated=bladder_inflated, antenna_active=transmitting)

    # Render HUD text inside header
    pill_color = EMERALD_ACCENT if ctd_active else (CYAN_ACCENT if not transmitting else (245, 158, 11))
    draw.rounded_rectangle([28, 19, 108, 39], radius=5, fill=pill_color)
    draw.text((36, 23), stage_pill, fill=(15, 23, 42), font=font_bold)
    
    draw.text((118, 20), stage_title, fill=WHITE, font=font_title)
    draw.text((28, 44), stage_detail, fill=MUTED, font=font_sm)
    
    draw.text((WIDTH - 145, 20), "10-DAY CYCLE", fill=CYAN_ACCENT, font=font_mono_bold)
    draw.text((WIDTH - 145, 42), f"Progress: {int((f+1)/FRAMES*100)}%", fill=MUTED, font=font_mono)

    frames.append(im)

out_path = r"c:\Users\ANKUSH\OneDrive\Desktop\ARGO AI\frontend\src\assets\argo-float-cycle.gif"
frames[0].save(
    out_path,
    save_all=True,
    append_images=frames[1:],
    duration=DURATION_MS,
    loop=0,
    optimize=False
)
print(f"Generated {out_path} ({len(frames)} frames, {os.path.getsize(out_path)} bytes)")
