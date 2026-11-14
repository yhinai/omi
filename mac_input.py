"""Small dependency-free Quartz adapter. Importing this module posts no input."""
import ctypes as C
import ctypes.util
import sys


class Point(C.Structure):
    _fields_ = [('x', C.c_double), ('y', C.c_double)]


class Size(C.Structure):
    _fields_ = [('width', C.c_double), ('height', C.c_double)]


class Rect(C.Structure):
    _fields_ = [('origin', Point), ('size', Size)]


class MacInput:
    def __init__(self):
        if sys.platform != 'darwin':
            raise RuntimeError('Mac controls require macOS')
        self.cg = C.CDLL(ctypes.util.find_library('CoreGraphics'))
        self.cf = C.CDLL(ctypes.util.find_library('CoreFoundation'))
        self.ax = C.CDLL(ctypes.util.find_library('ApplicationServices'))
        signatures = {
            'CGEventCreate': (C.c_void_p, [C.c_void_p]),
            'CGEventGetLocation': (Point, [C.c_void_p]),
            'CGEventCreateMouseEvent': (C.c_void_p, [C.c_void_p, C.c_uint32, Point, C.c_uint32]),
            'CGEventCreateScrollWheelEvent2': (C.c_void_p, [C.c_void_p, C.c_uint32, C.c_uint32,
                                                         C.c_int32, C.c_int32, C.c_int32]),
            'CGEventPost': (None, [C.c_uint32, C.c_void_p]),
            'CGEventSetFlags': (None, [C.c_void_p, C.c_uint64]),
            'CGEventSourceKeyState': (C.c_bool, [C.c_int32, C.c_uint16]),
            'CGEventSourceFlagsState': (C.c_uint64, [C.c_int32]),
            'CGGetActiveDisplayList': (C.c_int32, [C.c_uint32, C.POINTER(C.c_uint32), C.POINTER(C.c_uint32)]),
            'CGGetOnlineDisplayList': (C.c_int32, [C.c_uint32, C.POINTER(C.c_uint32), C.POINTER(C.c_uint32)]),
            'CGDisplayBounds': (Rect, [C.c_uint32]),
            'CGPreflightListenEventAccess': (C.c_bool, []),
        }
        for name, (result, arguments) in signatures.items():
            function = getattr(self.cg, name)
            function.restype, function.argtypes = result, arguments
        self.cf.CFRelease.argtypes = [C.c_void_p]
        self.cf.CFRelease.restype = None
        self.ax.AXIsProcessTrusted.restype = C.c_bool
        self.ax.AXIsProcessTrusted.argtypes = []

    def trusted(self):
        return bool(self.ax.AXIsProcessTrusted())

    def can_read_keys(self):
        return bool(self.cg.CGPreflightListenEventAccess())

    def key_down(self, code):
        return bool(self.cg.CGEventSourceKeyState(0, code))

    def control_chord(self):
        mask = (1 << 18) | (1 << 19) | (1 << 20)  # Control + Option + Command
        return self.cg.CGEventSourceFlagsState(0) & mask == mask

    def position(self):
        event = self.cg.CGEventCreate(None)
        if not event:
            raise RuntimeError('Cannot read pointer position')
        try:
            point = self.cg.CGEventGetLocation(event)
            return point.x, point.y
        finally:
            self.cf.CFRelease(event)

    def _post(self, event):
        if not event:
            raise RuntimeError('Cannot create Mac input event')
        try:
            self.cg.CGEventSetFlags(event, 0)
            self.cg.CGEventPost(0, event)
        finally:
            self.cf.CFRelease(event)

    def move(self, dx, dy):
        x, y = self.position()
        displays = (C.c_uint32 * 32)()
        count = C.c_uint32()
        error = self.cg.CGGetActiveDisplayList(32, displays, C.byref(count))
        if error or not count.value:
            error = self.cg.CGGetOnlineDisplayList(32, displays, C.byref(count))
        if error or not count.value:
            raise RuntimeError('Cannot read display bounds')
        # Clamp to the nearest actual display, including layouts with gaps.
        candidates = []
        for display in displays[:count.value]:
            r = self.cg.CGDisplayBounds(display)
            nx = max(r.origin.x, min(x + dx, r.origin.x + r.size.width - 1))
            ny = max(r.origin.y, min(y + dy, r.origin.y + r.size.height - 1))
            candidates.append(((nx-x-dx)**2 + (ny-y-dy)**2, nx, ny))
        _, nx, ny = min(candidates)
        self._post(self.cg.CGEventCreateMouseEvent(None, 5, Point(nx, ny), 0))

    def scroll(self, pixels):
        if pixels:
            self._post(self.cg.CGEventCreateScrollWheelEvent2(None, 0, 1, -int(pixels), 0, 0))

    def click(self):
        point = Point(*self.position())
        down = self.cg.CGEventCreateMouseEvent(None, 1, point, 0)
        up = self.cg.CGEventCreateMouseEvent(None, 2, point, 0)
        if not down or not up:
            for event in (down,up):
                if event: self.cf.CFRelease(event)
            raise RuntimeError('Cannot create Mac click events')
        self._post(down)
        self._post(up)
