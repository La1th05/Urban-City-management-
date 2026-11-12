import sys

from PySide6.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout,
                               QPushButton, QLabel, QTabWidget, QSpinBox,
                               QFormLayout, QButtonGroup, QFileDialog,
                               QMessageBox)
from PySide6.QtGui import (QIcon, QPainter, QColor, QPolygonF, QPixmap, QPen,
                           QPainterPath, QImage)
from PySide6.QtCore import Qt, QPointF

app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("Land Layout Planner")
window.setWindowIcon(QIcon("i.ico"))
window.setMinimumSize(1000, 600)
window.setStyleSheet("QWidget { background-color: #242424; color: #dddddd; }")

root = QHBoxLayout(window)
root.setContentsMargins(0, 0, 0, 0)
root.setSpacing(2)

# =========== SHARED STATE ===========
COLORS = {
    "Street": "#6b6b6b", "Green": "#2e9e4f", "House": "#e8b23a",
    "Building": "#d3543c", "Commercial": "#7d5bbe", "Water": "#3a7bd5",
    "Out of bound": "#111111", "Terrain": "#8a6f4d",
}
current_label = None
label_buttons = {}
view_mode = "pixels"
import_img = None   # the imported screenshot, kept for later use


# =========== TOOLS (5 stacked strips) ===========
tools = QWidget()
tools.setFixedWidth(200)
tools.setStyleSheet("background-color: #3a3a3a;")

tools_lay = QVBoxLayout(tools)
tools_lay.setContentsMargins(0, 0, 0, 0)
tools_lay.setSpacing(2)

def make_strip(height=None):
    strip = QWidget()
    strip.setStyleSheet("background-color: #2e2e2e;")
    lay = QVBoxLayout(strip)
    lay.setContentsMargins(6, 6, 6, 6)
    if height:
        strip.setFixedHeight(height)
    tools_lay.addWidget(strip, 0 if height else 1)
    return strip, lay

strip1, strip1_lay = make_strip(height=100)
strip2, strip2_lay = make_strip()
strip3, _ = make_strip()
strip4, _ = make_strip()
strip5, _ = make_strip()

logo = QLabel()
logo.setPixmap(QPixmap("i2.png"))
logo.setScaledContents(True)
logo.setFixedSize(100, 100)
strip1_lay.addWidget(logo, 0, Qt.AlignCenter)

# =========== STRIP 2 : DRAW / IMPORT ===========
title2 = QLabel("Draw / Import")
title2.setAlignment(Qt.AlignCenter)
title2.setStyleSheet("font-size: 13px; font-weight: bold; color: white;")
strip2_lay.addWidget(title2)

tabs = QTabWidget()
tabs.setStyleSheet("""
    QTabWidget::pane     { border: 1px solid #555; }
    QTabBar::tab         { background: #3a3a3a; color: #bbb;
                           padding: 3px 6px; font-size: 11px; }
    QTabBar::tab:selected { background: #555555; color: white; }
""")

tab_import = QWidget()
imp_lay = QVBoxLayout(tab_import)
imp_lay.setContentsMargins(6, 6, 6, 6)
btn_import = QPushButton("Import Image")
imp_lay.addWidget(btn_import)

import_preview = QLabel("No image yet")
import_preview.setAlignment(Qt.AlignCenter)
import_preview.setFixedHeight(140)
import_preview.setStyleSheet("border: 1px solid #555;")
imp_lay.addWidget(import_preview)
imp_lay.addStretch(1)

tab_size = QWidget()
form_size = QFormLayout(tab_size)
form_size.setContentsMargins(6, 6, 6, 6)
spin_w     = QSpinBox(); spin_w.setRange(1, 10000);  spin_w.setValue(100)
spin_h     = QSpinBox(); spin_h.setRange(1, 10000);  spin_h.setValue(100)
spin_mpp   = QSpinBox(); spin_mpp.setRange(1, 100);  spin_mpp.setValue(2)
spin_brush = QSpinBox(); spin_brush.setRange(1, 25); spin_brush.setValue(1)
spin_gap   = QSpinBox(); spin_gap.setRange(0, 10);   spin_gap.setValue(0)
form_size.addRow("Width", spin_w)
form_size.addRow("Height", spin_h)
form_size.addRow("Meters / Cell", spin_mpp)
form_size.addRow("Brush Size", spin_brush)
form_size.addRow("Cell Gap (px)", spin_gap)

tab_labels = QWidget()
lab_lay = QVBoxLayout(tab_labels)
lab_lay.setContentsMargins(6, 6, 6, 6)
lab_lay.setSpacing(4)

def legend_header(text):
    h = QLabel(text)
    h.setStyleSheet("font-weight: bold; color: white; font-size: 11px;")
    lab_lay.addWidget(h)

def legend_row(text):
    row = QWidget()
    rh = QHBoxLayout(row)
    rh.setContentsMargins(0, 0, 0, 0)
    sw = QPushButton()
    sw.setFixedSize(14, 14)
    sw.setCursor(Qt.PointingHandCursor)
    sw.clicked.connect(lambda _, n=text: select_label(n))
    label_buttons[text] = sw
    rh.addWidget(sw)
    rh.addWidget(QLabel(text))
    rh.addStretch(1)
    lab_lay.addWidget(row)

def select_label(name):
    global current_label
    current_label = name
    for n, b in label_buttons.items():
        border = "2px solid white" if n == name else "1px solid #666"
        b.setStyleSheet(f"background-color: {COLORS[n]}; border: {border};")

legend_header("Land")
legend_row("Street")
legend_row("Green")
legend_row("House")
legend_row("Building")
legend_row("Commercial")
legend_header("Water")
legend_row("Water")
legend_header("Out of bounds")
legend_row("Out of bound")
legend_row("Terrain")

select_label("Street")
lab_lay.addStretch(1)

tab_apply = QWidget()
apply_lay = QVBoxLayout(tab_apply)
apply_lay.setContentsMargins(6, 6, 6, 6)

btn_apply = QPushButton("Apply")
btn_apply.setStyleSheet("""
    QPushButton { background-color: #2e9e4f; color: white;
                  font-weight: bold; padding: 6px; border: none; }
    QPushButton:hover { background-color: #3cb861; }
""")
apply_lay.addWidget(btn_apply)

btn_clear = QPushButton("Clear Painting")
btn_clear.setStyleSheet("""
    QPushButton { background-color: #c0392b; color: white;
                  font-weight: bold; padding: 6px; border: none; }
    QPushButton:hover { background-color: #e74c3c; }
""")
apply_lay.addWidget(btn_clear)
apply_lay.addStretch(1)

tabs.addTab(tab_import, "Import")
tabs.addTab(tab_size, "Size")
tabs.addTab(tab_labels, "Labels")
tabs.addTab(tab_apply, "Apply")

strip2_lay.addWidget(tabs)

# =========== GRID CANVAS ===========
class GridCanvas(QWidget):
    """Pixel grid + vectorization: 4-sided regions stay razor sharp,
    everything organic is Taubin-smoothed into curves."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = 0
        self.cols = 0
        self.cells = []
        self.painting = False
        self.paint_value = None
        self.zoom = 1.0
        self.brush = 1
        self.spacing = 0
        self.hover_cell = None
        self.vec_paths = []       # cached [(label, QPainterPath), ...]
        self._vec_dirty = True
        self.setMouseTracking(True)

    def set_grid(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.cells = [[None] * cols for _ in range(rows)]
        self.zoom = 1.0
        self.hover_cell = None
        self._vec_dirty = True
        self.update()

    def load_grid(self, cells):
        """Adopt a ready-made grid (rows of labels / None) built outside,
        e.g. by the image importer."""
        self.rows = len(cells)
        self.cols = len(cells[0]) if self.rows else 0
        self.cells = cells
        self.zoom = 1.0
        self.hover_cell = None
        self._vec_dirty = True
        self.update()

    def clear(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.cells[r][c] = None
        self._vec_dirty = True
        self.update()

    # ---------- geometry ----------
    def _layout(self):
        if not self.cells:
            return 10.0, 0.0, 0.0
        fit = min(self.width() / self.cols, self.height() / self.rows)
        cell = fit * self.zoom
        ox = (self.width() - self.cols * cell) / 2
        oy = (self.height() - self.rows * cell) / 2
        return cell, ox, oy

    # ---------- VECTORIZATION PIPELINE ----------
    def build_vector_paths(self):
        """Same-label regions -> traced outlines. Loops with 3-8 true corners
        and no short edges are kept pixel-sharp; all others are
        Taubin-smoothed. Coordinates in CELL units; zoom applied at draw."""
        self.vec_paths = []
        if not self.cells:
            return
        seen = [[False] * self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                label = self.cells[r][c]
                if label is None or seen[r][c]:
                    continue
                region = self._flood(r, c, label, seen)
                for loop in self._trace_region(region):
                    verts = self._simplify(loop)
                    edges_long = all(
                        abs(verts[i][0] - verts[(i + 1) % len(verts)][0])
                        + abs(verts[i][1] - verts[(i + 1) % len(verts)][1]) >= 2
                        for i in range(len(verts))
                    )
                    if 3 <= len(verts) <= 8 and edges_long:
                        final = loop
                    else:
                        final = self._smooth(loop)
                    path = QPainterPath()
                    path.moveTo(QPointF(*final[0]))
                    for pt in final[1:]:
                        path.lineTo(QPointF(*pt))
                    path.closeSubpath()
                    self.vec_paths.append((label, path))

    def _flood(self, r0, c0, label, seen):
        """Collect the 4-connected blob of `label` containing (r0, c0)."""
        stack = [(r0, c0)]
        seen[r0][c0] = True
        region = []
        while stack:
            r, c = stack.pop()
            region.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if (0 <= rr < self.rows and 0 <= cc < self.cols
                        and not seen[rr][cc] and self.cells[rr][cc] == label):
                    seen[rr][cc] = True
                    stack.append((rr, cc))
        return region

    def _trace_region(self, region):
        """Chain the region's boundary edges into closed loops (cell units)."""
        rset = set(region)
        edges = {}
        def add(a, b):
            edges.setdefault(a, []).append(b)
        for (r, c) in region:
            if (r - 1, c) not in rset: add((c, r),         (c + 1, r))      # top
            if (r + 1, c) not in rset: add((c + 1, r + 1), (c, r + 1))      # bottom
            if (r, c - 1) not in rset: add((c, r + 1),     (c, r))          # left
            if (r, c + 1) not in rset: add((c + 1, r),     (c + 1, r + 1))  # right

        loops = []
        while edges:
            start = next(iter(edges))
            cur = start
            loop = [start]
            while True:
                ends = edges[cur]
                nxt = ends.pop()
                if not ends:
                    del edges[cur]         # retire the corner we are LEAVING
                cur = nxt
                if cur == start:
                    break
                loop.append(cur)
            loops.append(loop)
        return loops

    def _simplify(self, pts):
        """Merge collinear runs: return only the TRUE corner vertices.
        A traced rectangle collapses to exactly 4 points; a staircase
        keeps a vertex per step (direction changes constantly)."""
        n = len(pts)
        if n < 4:
            return list(pts)
        def direction(a, b):
            if b[0] != a[0]:
                return (1 if b[0] > a[0] else -1, 0)
            return (0, 1 if b[1] > a[1] else -1)
        verts = []
        for i in range(n):
            if direction(pts[i - 1], pts[i]) != direction(pts[i], pts[(i + 1) % n]):
                verts.append(pts[i])
        return verts

    def _smooth(self, pts):
        """Taubin smoothing — the true-circle maker.

        Each pass pulls every point toward the midpoint of its two neighbours
        (factor lam), then a second pass pushes it slightly back out (factor
        mu, negative). Pixel-scale wobble is high-frequency and dies in a few
        iterations; the overall shape barely moves, so a pixelated circle
        relaxes into a true circle and a staircase line into a capsule.
        Iteration count scales with outline size so tiny blobs don't vanish."""
        n = len(pts)
        if n < 4:
            return pts
        lam, mu = 0.5, -0.53
        iters = max(2, min(12, n // 4))
        for _ in range(iters):
            for factor in (lam, mu):
                out = []
                for i in range(n):
                    ax, ay = pts[i]
                    bx, by = pts[i - 1]           # previous (wraps at 0)
                    cx, cy = pts[(i + 1) % n]     # next
                    mx, my = (bx + cx) / 2, (by + cy) / 2
                    out.append((ax + factor * (mx - ax),
                                ay + factor * (my - ay)))
                pts = out
        return pts

    # ---------- rendering ----------
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#242424"))
        if not self.cells:
            return
        cell, ox, oy = self._layout()

        if view_mode == "vectors":
            if self._vec_dirty:
                self.build_vector_paths()
                self._vec_dirty = False
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(Qt.NoPen)
            p.save()
            p.translate(ox, oy)
            p.scale(cell, cell)
            for label, path in self.vec_paths:
                col = QColor(COLORS[label])
                p.fillPath(path, col)
                # hairline stroke of the same color hides seams where
                # two smoothed regions share a boundary
                p.strokePath(path, QPen(col, 0.12))
            p.restore()
            return

        # ---------- pixels mode ----------
        gap = self.spacing
        side = cell - gap if gap > 0 else cell + 1
        side = max(side, 1)
        inset = gap / 2
        for r in range(self.rows):
            for c in range(self.cols):
                label = self.cells[r][c]
                if label:
                    p.fillRect(int(ox + c * cell + inset),
                               int(oy + r * cell + inset),
                               int(side), int(side),
                               QColor(COLORS[label]))
        if gap == 0:
            p.setPen(QPen(QColor("#1a1a1a")))
            gw, gh = self.cols * cell, self.rows * cell
            for c in range(self.cols + 1):
                x = int(ox + c * cell); p.drawLine(x, int(oy), x, int(oy + gh))
            for r in range(self.rows + 1):
                y = int(oy + r * cell); p.drawLine(int(ox), y, int(ox + gw), y)

        if self.hover_cell:
            hr, hc = self.hover_cell
            half = self.brush // 2
            p.setPen(QPen(QColor("white")))
            p.setBrush(Qt.NoBrush)
            p.drawRect(int(ox + (hc - half) * cell),
                       int(oy + (hr - half) * cell),
                       int(self.brush * cell),
                       int(self.brush * cell))

    # ---------- wheel zoom ----------
    def wheelEvent(self, event):
        if not self.cells:
            return
        delta = event.angleDelta().y()
        factor = 1.25 if delta > 0 else 0.8
        new_zoom = self.zoom * factor
        if self._fit_check(new_zoom):
            self.zoom = new_zoom
            self.update()

    def _fit_check(self, zoom):
        fit = min(self.width() / self.cols, self.height() / self.rows)
        return 2.0 <= fit * zoom <= 80.0

    # ---------- mouse ----------
    def _cell_at(self, pos):
        if not self.cells:
            return None
        cell, ox, oy = self._layout()
        c = int((pos.x() - ox) // cell)
        r = int((pos.y() - oy) // cell)
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return r, c
        return None

    def mousePressEvent(self, event):
        cell = self._cell_at(event.position())
        if not cell:
            return
        self.painting = True
        self.paint_value = None if event.button() == Qt.RightButton else current_label
        self._paint_cell(cell)

    def mouseMoveEvent(self, event):
        cell = self._cell_at(event.position())
        if view_mode == "pixels" and cell != self.hover_cell:
            self.hover_cell = cell
            self.update()
        if self.painting and cell:
            self._paint_cell(cell)

    def leaveEvent(self, event):
        self.hover_cell = None
        self.update()

    def mouseReleaseEvent(self, event):
        self.painting = False

    def _paint_cell(self, cell_rc):
        r, c = cell_rc
        half = self.brush // 2
        changed = False
        for rr in range(r - half, r - half + self.brush):
            for cc in range(c - half, c - half + self.brush):
                if 0 <= rr < self.rows and 0 <= cc < self.cols:
                    if self.cells[rr][cc] != self.paint_value:
                        self.cells[rr][cc] = self.paint_value
                        changed = True
        if changed:
            self._vec_dirty = True
            self.update()

# =========== CANVAS AREA (view bar + canvas) ===========
canvas_area = QWidget()
canvas_v = QVBoxLayout(canvas_area)
canvas_v.setContentsMargins(0, 0, 0, 0)
canvas_v.setSpacing(2)

view_bar = QWidget()
view_bar.setFixedHeight(28)
view_h = QHBoxLayout(view_bar)
view_h.setContentsMargins(6, 0, 6, 0)
view_h.addStretch(1)

btn_vectors = QPushButton("Vectors")
btn_pixels  = QPushButton("Pixels")
for b in (btn_vectors, btn_pixels):
    b.setCheckable(True)
    b.setCursor(Qt.PointingHandCursor)
    b.setStyleSheet("""
        QPushButton { background: #3a3a3a; color: #bbb; border: none;
                      padding: 2px 10px; font-size: 11px; }
        QPushButton:checked { background: #2e9e4f; color: white;
                              font-weight: bold; }
        QPushButton:hover:!checked { background: #4a4a4a; }
    """)

btn_pixels.setChecked(True)

view_group = QButtonGroup(window)
view_group.addButton(btn_vectors)
view_group.addButton(btn_pixels)

view_h.addWidget(btn_vectors)
view_h.addWidget(btn_pixels)

canvas_v.addWidget(view_bar)

canvas = GridCanvas()
canvas_v.addWidget(canvas, 1)

def set_view_mode():
    global view_mode
    view_mode = "vectors" if btn_vectors.isChecked() else "pixels"
    canvas.update()

btn_vectors.clicked.connect(set_view_mode)
btn_pixels.clicked.connect(set_view_mode)

# =========== ANALYTICS (collapsible) ===========
analytics = QWidget()
analytics.setFixedWidth(200)
analytics.setStyleSheet("background-color: #3a3a3a;")


class ToggleTab(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.open_state = True
        self.hovered = False
        self.color = QColor("#3a3a3a")
        self.hover_color = QColor("#555555")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(20, 100)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(self.hover_color if self.hovered else self.color)
        p.drawRoundedRect(self.rect().adjusted(0, 0, 10, 0), 10, 10)
        self.draw_arrow(p)

    def draw_arrow(self, p):
        p.setBrush(Qt.white)
        cx, cy = self.width() / 2, self.height() / 2
        s = 7
        if self.open_state:
            p.drawPolygon(QPolygonF([QPointF(cx - s, cy - s),
                                     QPointF(cx - s, cy + s),
                                     QPointF(cx + s, cy)]))
        else:
            p.drawPolygon(QPolygonF([QPointF(cx + s, cy - s),
                                     QPointF(cx + s, cy + s),
                                     QPointF(cx - s, cy)]))

    def enterEvent(self, event):
        self.hovered = True
        self.update()

    def leaveEvent(self, event):
        self.hovered = False
        self.update()


toggle = ToggleTab()

root.addWidget(tools)
root.addWidget(canvas_area, 1)
root.addWidget(toggle)
root.addWidget(analytics)

# =========== WIRING ===========
def toggle_analytics():
    analytics.setVisible(not analytics.isVisible())
    toggle.open_state = analytics.isVisible()
    toggle.update()

toggle.clicked.connect(toggle_analytics)

def apply_size():
    canvas.set_grid(spin_h.value(), spin_w.value())

btn_apply.clicked.connect(apply_size)
btn_clear.clicked.connect(canvas.clear)

def set_brush(v):
    canvas.brush = v

spin_brush.valueChanged.connect(set_brush)

def set_gap(v):
    canvas.spacing = v
    canvas.update()

spin_gap.valueChanged.connect(set_gap)

# =========== IMAGE IMPORT (Google Maps map view) ===========
# boundary: a BLACK line you draw around the area
# inside:   roads white/yellow/orange/red -> Street, parks -> Green,
#           water -> Blue, anything undecided -> Terrain brown
# the image is AUTO-CROPPED to your area first, so the whole cell budget
# is spent on the land, not on the out-of-bound margin around it

IMPORT_MAX_SIDE = 300   # the grid never gets bigger than this. 300 is a
                        # SPEED budget, not a wall (painting loops over every
                        # cell). After the auto-crop it covers only your land,
                        # which is worth roughly 2x the detail it was before.
                        # Tiny roads still? raise to 400 — import and painting
                        # get somewhat slower.
DETECT_MAX_SIDE = 800   # resolution used to scan pixels
LINE_MAX = 40           # a pixel is "boundary line" when max(r,g,b) <= this.
                        # Pen black is 0-30; Google's name labels are ~90+
                        # gray, so they stay out.
                        # Line comes out patchy? raise to 60 and save as PNG.
                        # Letters/icons turning black? lower to 30.

# ---- color knobs (map view) ----
HUE_GREEN = (75, 170)    # parks/trees hue window. Starts at 75 on purpose:
                         # Google's yellow / yellowish road tints sit below
                         # ~65 hue and must NOT vote Green.
                         # Real parks vanish? lower to 70.
HUE_WATER = (170, 260)   # hue window for water
SAT_MIN = 25             # green/water must be at least this colorful
ROAD_HUE_MAX = 65        # colored roads = hue 330..360 + 0..65 (red..yellow)
SAT_ROAD = 50            # colored roads need real color (beige land is ~10)
ROAD_VAL_MIN = 140       # roads are bright; dark grays never count
ROAD_WHITE_VAL = 252     # white roads: NEAR-pure white only (road fill is
                         # 255). Was 248 and gray blocks leaked in — raised
                         # to narrow the band. Streets missing now? try 250.
ROAD_WHITE_SAT = 20      # ...and even more colorless than before (was 25)
VOTE_MIN_PCT = 0.5       # a label needs this fraction of the cell's scan
                         # pixels to win the cell (majority)
VOTE_RATIO = 2           # winner needs this many times the runner-up
CROP_MARGIN_CELLS = 10   # how much land-margin the auto-crop keeps around
                         # the pocket (covers the line's own thickness)


def pixel_label(r, g, b):
    """One scan pixel -> 'Water' / 'Green' / 'Street' / None.

    Google Maps DRAWINGS use flat, agreed-on colors — no sun, no shadow —
    so this is mostly exact color families:
      roads:   white / yellow / orange / red (by road importance)
      parks:   green
      water:   blue
      land, buildings, name labels -> light warm gray -> None (paintable)
    Hue = position on the 0-360 color wheel (which family), sat = how
    colorful, val = brightness."""
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    if d == 0:
        h = 0.0
    elif mx == r:
        h = 60 * ((g - b) / d)
    elif mx == g:
        h = 60 * (2 + (b - r) / d)
    else:
        h = 60 * (4 + (r - g) / d)
    if h < 0:
        h += 360
    s = 255 * d / mx if mx else 0
    v = mx
    if HUE_WATER[0] <= h <= HUE_WATER[1] and s >= SAT_MIN and v >= 120:
        return "Water"
    if HUE_GREEN[0] <= h <= HUE_GREEN[1] and s >= SAT_MIN and v >= 100:
        return "Green"
    warm = h <= ROAD_HUE_MAX or h >= 330        # the red..yellow band
    if warm and s >= SAT_ROAD and v >= ROAD_VAL_MIN:
        return "Street"                          # colored roads
    if s <= ROAD_WHITE_SAT and v >= ROAD_WHITE_VAL:
        return "Street"                          # white roads
    return None


def classify_import(original, _crop=True):
    """Screenshot -> grid rows using the drawn BLACK line.
    The BIGGEST pocket closed by the line is the land; the BIGGEST dark
    component is the boundary wall; dark specks inside are demoted to land.
    The image is first AUTO-CROPPED to the pocket's bounding box and the
    whole detection re-runs on the crop, so the 300-cell budget is spent
    on the land instead of the out-of-bound margin. Land cells vote
    Water/Green/Street; undecided land becomes Terrain; letter-cuts in
    roads are bridged and 1-wide diagonal roads are stitched.
    Returns the grid, or None (after explaining why in a popup)."""
    # --- two sizes from one idea: k = cap / longest side ---
    k = min(1.0, IMPORT_MAX_SIDE / max(original.width(), original.height()))
    gw = max(1, round(original.width() * k))
    gh = max(1, round(original.height() * k))
    dk = min(1.0, DETECT_MAX_SIDE / max(original.width(), original.height()))
    detect = original.scaled(max(1, round(original.width() * dk)),
                             max(1, round(original.height() * dk)),
                             Qt.IgnoreAspectRatio, Qt.FastTransformation
                             ).convertToFormat(QImage.Format_ARGB32)
    dw, dh = detect.width(), detect.height()

    # --- 1) find BLACK pixels and max-pool them into grid cells ---
    line = [[False] * gw for _ in range(gh)]
    buf = detect.constBits().tobytes()   # raw bytes, BGRA per pixel
    bpr = detect.bytesPerLine()
    line_found = 0
    for dy in range(dh):
        base = dy * bpr
        lrow = line[dy * gh // dh]
        for dx in range(dw):
            o = base + dx * 4
            if (buf[o + 2] <= LINE_MAX and buf[o + 1] <= LINE_MAX
                    and buf[o] <= LINE_MAX):
                lrow[dx * gw // dw] = True
                line_found += 1
    if line_found == 0:
        QMessageBox.warning(
            window, "Import",
            "No black line found in this image.\n"
            "Draw a black line around your area in Google Maps and\n"
            "screenshot it with the whole line visible.")
        return None

    # --- 2) thicken the line by one cell ---
    grown = [row[:] for row in line]
    for r in range(gh):
        for c in range(gw):
            if not line[r][c] and (
                    (r > 0 and line[r - 1][c])
                    or (r < gh - 1 and line[r + 1][c])
                    or (c > 0 and line[r][c - 1])
                    or (c < gw - 1 and line[r][c + 1])):
                grown[r][c] = True
    line = grown

    # --- 3) flood from the image borders over non-line cells = outside ---
    outside = [[False] * gw for _ in range(gh)]
    stack = [(0, c) for c in range(gw)] + [(gh - 1, c) for c in range(gw)]
    stack += [(r, 0) for r in range(gh)] + [(r, gw - 1) for r in range(gh)]
    while stack:
        r, c = stack.pop()
        if (0 <= r < gh and 0 <= c < gw and not outside[r][c]
                and not line[r][c]):
            outside[r][c] = True
            stack.append((r + 1, c))
            stack.append((r - 1, c))
            stack.append((r, c + 1))
            stack.append((r, c - 1))

    # --- 4) whatever the flood could NOT reach = trapped behind some loop ---
    inside = [[not line[r][c] and not outside[r][c] for c in range(gw)]
              for r in range(gh)]

    # --- 5) keep ONLY the biggest trapped pocket = your area ---
    seen = [[False] * gw for _ in range(gh)]
    best = []
    for r in range(gh):
        for c in range(gw):
            if not inside[r][c] or seen[r][c]:
                continue
            comp = []
            stack = [(r, c)]
            seen[r][c] = True
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc),
                               (cr, cc - 1), (cr, cc + 1)):
                    if (0 <= nr < gh and 0 <= nc < gw
                            and inside[nr][nc] and not seen[nr][nc]):
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            if len(comp) > len(best):
                best = comp
    if len(best) < max(8, gw * gh // 2000):
        QMessageBox.warning(
            window, "Import",
            "A black line was found, but no big closed loop came out of it.\n"
            "The line is probably open or cut off by the image edge.\n"
            "Close the loop fully inside the screenshot and try again.")
        return None

    # --- 5b) NEW: AUTO-CROP to the land and re-run on it ---
    # The cell budget was spent on the WHOLE screenshot, but the out-of-bound
    # margin around the line is usually a big part of the picture and is
    # dead weight. Crop the original to the pocket's bounding box (+ a small
    # margin for the line itself) and run the whole detection again: the
    # same 300 cells now cover only the land, so roads come out wider.
    # This is exactly "use the empty out-of-bound space to enlarge the
    # image" — the margin's cells are given back to the land.
    if _crop:
        rs = [r for r, _ in best]
        cs = [c for _, c in best]
        top, bot = min(rs), max(rs)
        left, right = min(cs), max(cs)
        gains = ((bot - top + 1) < gh * 0.95
                 or (right - left + 1) < gw * 0.95)
        if gains:
            m = round(CROP_MARGIN_CELLS / k)      # margin back in image px
            x0 = max(0, int(left / k) - m)
            y0 = max(0, int(top / k) - m)
            x1 = min(original.width(), int((right + 1) / k) + m)
            y1 = min(original.height(), int((bot + 1) / k) + m)
            if x1 - x0 > 5 and y1 - y0 > 5:
                return classify_import(original.copy(x0, y0, x1 - x0, y1 - y0),
                                       _crop=False)

    # --- 6) the wall is only the BIGGEST dark component ---
    # your drawn loop is one huge connected stroke; one-way arrows, POI
    # icons and other dark specks are small separate components. Flood all
    # of them (same pattern again), keep the largest as the boundary wall,
    # and DEMOTE every small component that touches the pocket to plain
    # land — so arrows inside your area get voted and filled, not blacked.
    seen_l = [[False] * gw for _ in range(gh)]
    comps = []
    for r in range(gh):
        for c in range(gw):
            if not line[r][c] or seen_l[r][c]:
                continue
            comp = []
            stack = [(r, c)]
            seen_l[r][c] = True
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc),
                               (cr, cc - 1), (cr, cc + 1)):
                    if (0 <= nr < gh and 0 <= nc < gw
                            and line[nr][nc] and not seen_l[nr][nc]):
                        seen_l[nr][nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    main_i = max(range(len(comps)), key=lambda i: len(comps[i]))

    is_inside = [[False] * gw for _ in range(gh)]
    for (r, c) in best:
        is_inside[r][c] = True
    for i, comp in enumerate(comps):
        if i == main_i:
            continue
        touches = any(
            any(0 <= nr < gh and 0 <= nc < gw and is_inside[nr][nc]
                for nr, nc in ((r - 1, c), (r + 1, c),
                               (r, c - 1), (r, c + 1)))
            for (r, c) in comp)
        if touches:
            for (r, c) in comp:
                is_inside[r][c] = True

    # --- 7) color vote inside the land ---
    # every scan pixel votes for its cell's label; only pixels whose cell is
    # land get classified. Street-name letters are thin gray text: their
    # pixels vote nothing, the road pixels around them vote Street.
    ppc = (dw * dh) / (gw * gh)
    vote_min = max(1, round(ppc * VOTE_MIN_PCT))

    votes_w = [[0] * gw for _ in range(gh)]
    votes_g = [[0] * gw for _ in range(gh)]
    votes_s = [[0] * gw for _ in range(gh)]
    for dy in range(dh):
        base = dy * bpr
        grow = dy * gh // dh
        for dx in range(dw):
            ccol = dx * gw // dw
            if not is_inside[grow][ccol]:
                continue
            o = base + dx * 4
            lab = pixel_label(buf[o + 2], buf[o + 1], buf[o])
            if lab == "Water":
                votes_w[grow][ccol] += 1
            elif lab == "Green":
                votes_g[grow][ccol] += 1
            elif lab == "Street":
                votes_s[grow][ccol] += 1

    # --- 8) final grid: land cells get their winning label if decisive,
    #     undecided land = Terrain; everything else -> 'Out of bound' ---
    grid = []
    for r in range(gh):
        row = []
        for c in range(gw):
            if not is_inside[r][c]:
                row.append("Out of bound")
                continue
            pairs = sorted((("Water", votes_w[r][c]),
                            ("Green", votes_g[r][c]),
                            ("Street", votes_s[r][c])),
                           key=lambda t: t[1], reverse=True)
            (name1, n1), (_, n2) = pairs[0], pairs[1]
            if n1 >= vote_min and n1 >= VOTE_RATIO * n2:
                row.append(name1)
            else:
                row.append("Terrain")
        grid.append(row)

    # --- 9) close the cuts letters and arrows punched into roads ---
    # BRIDGE TEST: promote a Terrain cell to Street only when two OPPOSITE
    # neighbours are both Street (left+right, up+down, or either diagonal
    # pair). A letter cut sits ON the road, so the road continues on both
    # sides of it -> filled. A cell merely NEXT to a road never has road on
    # opposite sides -> never promoted, so roads keep their true width.
    def st(r0, c0):
        return 0 <= r0 < gh and 0 <= c0 < gw and grid[r0][c0] == "Street"

    def bridge_fill():
        for _ in range(3):
            promo = []
            for r in range(gh):
                for c in range(gw):
                    if not is_inside[r][c] or grid[r][c] != "Terrain":
                        continue
                    if ((st(r, c - 1) and st(r, c + 1))
                            or (st(r - 1, c) and st(r + 1, c))
                            or (st(r - 1, c - 1) and st(r + 1, c + 1))
                            or (st(r - 1, c + 1) and st(r + 1, c - 1))):
                        promo.append((r, c))
            if not promo:
                break
            for r, c in promo:
                grid[r][c] = "Street"

    bridge_fill()

    # --- 10) NEW: stitch 1-cell-wide diagonal roads ---
    # A diagonal road only 1 cell wide is a dashed line: its cells touch at
    # CORNERS only, and corner-touching cells are not connected (streets
    # use 4-neighbour / edge connectivity). Wherever two Street cells touch
    # diagonally and both shared neighbours are still Terrain, promote one
    # of them — the pair becomes an edge-connected staircase, which vector
    # mode then smooths into a clean straight diagonal band.
    # Straight roads never trigger this (their cells touch edge-to-edge),
    # and 2-wide diagonals already have Street shared neighbours, so
    # nothing is fattened — only true corner-gaps get bridged.
    for r in range(gh - 1):
        for c in range(gw - 1):
            tl = grid[r][c] == "Street"
            tr = grid[r][c + 1] == "Street"
            bl = grid[r + 1][c] == "Street"
            br = grid[r + 1][c + 1] == "Street"
            if tl and br and not tr and not bl:
                if grid[r][c + 1] == "Terrain":
                    grid[r][c + 1] = "Street"
                elif grid[r + 1][c] == "Terrain":
                    grid[r + 1][c] = "Street"
            elif bl and tr and not tl and not br:
                if grid[r + 1][c] == "Terrain":
                    grid[r + 1][c] = "Street"
                elif grid[r][c + 1] == "Terrain":
                    grid[r][c + 1] = "Street"
    bridge_fill()   # the new stitches can unlock a few more letter cuts

    return grid


def import_image():
    global import_img
    path, _ = QFileDialog.getOpenFileName(
        window, "Choose image", "",
        "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
    if not path:
        return
    img = QImage(path)
    if img.isNull():
        QMessageBox.warning(window, "Import", "Could not load that image.")
        return
    import_img = img
    import_preview.setPixmap(
        QPixmap.fromImage(img).scaled(180, 140, Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation))
    cells = classify_import(img)
    if cells is None:
        return
    canvas.load_grid(cells)
    gh, gw = len(cells), len(cells[0])
    spin_w.setValue(gw)
    spin_h.setValue(gh)
    import_preview.setToolTip(
        f"{img.width()} x {img.height()} px  ->  grid {gw} x {gh}")


btn_import.clicked.connect(import_image)

window.showMaximized()
sys.exit(app.exec())