from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import shapely.geometry as sg

@dataclass
class DesignBlock:
    z_start: float
    z_end: float
    batter_angle_deg: float
    berm_width: float

@dataclass
class PitDesignParams:
    bench_height: float
    batter_angle_deg: float
    berm_width: float
    target_elevation: float
    design_direction: str = "Downward"  # "Downward" or "Upward"
    variable_params: List[DesignBlock] = field(default_factory=list)

@dataclass
class Mesh3D:
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    faces: List[Tuple[int, int, int]] = field(default_factory=list)

    def extend(self, other: 'Mesh3D'):
        offset = len(self.vertices)
        self.vertices.extend(other.vertices)
        for f in other.faces:
            self.faces.append((f[0] + offset, f[1] + offset, f[2] + offset))

@dataclass
class BenchGeometry:
    bench_id: int
    z_crest: float
    z_toe: float
    crest_polys: List[sg.Polygon] = field(default_factory=list)
    toe_polys: List[sg.Polygon] = field(default_factory=list)
    face_mesh: Mesh3D = field(default_factory=Mesh3D)
    berm_mesh: Mesh3D = field(default_factory=Mesh3D)
    diagnostics: List[str] = field(default_factory=list)
