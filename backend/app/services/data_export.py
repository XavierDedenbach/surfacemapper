import os
import json
import pandas as pd
from typing import Any, Optional
from app.models.data_models import AnalysisResults, StatisticalAnalysis

class DataExporter:
    def export_analysis_results(self, results: AnalysisResults, fmt: str, file_path: str, metadata: Optional[dict] = None) -> str:
        fmt = fmt.lower()
        if not file_path or not os.path.isdir(os.path.dirname(file_path)):
            raise ValueError("Invalid file path")
        if fmt == "csv":
            df = pd.DataFrame([vars(v) for v in results.volume_results])
            df.to_csv(file_path, index=False)
        elif fmt == "json":
            with open(file_path, "w") as f:
                json.dump(results.model_dump(), f, indent=2)
        elif fmt == "excel":
            with pd.ExcelWriter(file_path) as writer:
                pd.DataFrame([vars(v) for v in results.volume_results]).to_excel(writer, sheet_name="Volume", index=False)
                pd.DataFrame([vars(t) for t in results.thickness_results]).to_excel(writer, sheet_name="Thickness", index=False)
                pd.DataFrame([vars(c) for c in results.compaction_results]).to_excel(writer, sheet_name="Compaction", index=False)
        else:
            raise ValueError(f"Unsupported export format: {fmt}")
        return file_path

    def export_statistical_data(self, stats: StatisticalAnalysis, fmt: str, file_path: str) -> str:
        fmt = fmt.lower()
        if not file_path or not os.path.isdir(os.path.dirname(file_path)):
            raise ValueError("Invalid file path")
        if fmt == "csv":
            df = pd.DataFrame([stats.model_dump()])
            df.to_csv(file_path, index=False)
        elif fmt == "json":
            with open(file_path, "w") as f:
                json.dump(stats.model_dump(), f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {fmt}")
        return file_path

    def export_surface_data(self, surface_data: dict, fmt: str, file_path: str) -> str:
        fmt = fmt.lower()
        if not file_path or not os.path.isdir(os.path.dirname(file_path)):
            raise ValueError("Invalid file path")
        vertices = surface_data.get("vertices", [])
        faces = surface_data.get("faces", [])
        if fmt == "ply":
            with open(file_path, "w") as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {len(vertices)}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write(f"element face {len(faces)}\n")
                f.write("property list uchar int vertex_indices\nend_header\n")
                for v in vertices:
                    f.write(f"{v[0]} {v[1]} {v[2]}\n")
                for face in faces:
                    f.write(f"3 {' '.join(str(idx) for idx in face)}\n")
        elif fmt == "obj":
            with open(file_path, "w") as f:
                for v in vertices:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in faces:
                    f.write(f"f {' '.join(str(idx+1) for idx in face)}\n")
        elif fmt == "stl":
            with open(file_path, "w") as f:
                f.write("solid surface\n")
                for face in faces:
                    f.write("facet normal 0 0 0\nouter loop\n")
                    for idx in face:
                        v = vertices[idx]
                        f.write(f"vertex {v[0]} {v[1]} {v[2]}\n")
                    f.write("endloop\nendfacet\n")
                f.write("endsolid surface\n")
        elif fmt == "xyz":
            lines = []
            for v in vertices:
                if len(v) == 3:
                    try:
                        line = f"{float(v[0])} {float(v[1])} {float(v[2])}"
                        lines.append(line)
                    except Exception:
                        continue
            with open(file_path, "w") as f:
                f.write("\n".join(lines))
        else:
            raise ValueError(f"Unsupported export format: {fmt}")
        return file_path 