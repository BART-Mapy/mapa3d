#!/usr/bin/env python3
"""Spatial join building outlines with building:part and compute 3D volumes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Any, Iterable

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import unary_union

FLOOR_HEIGHT = 4.0
MAX_BUILDING_LEVELS = 25


@dataclass
class BuildingFeature:
    osm_id: str
    tags: dict[str, str]
    geometry: Polygon
    inner_rings: list[Polygon]
    is_part: bool


def parse_level_value(raw: str | None) -> int | None:
    if not raw:
        return None
    token = raw.strip().split(";")[0].strip()
    try:
        value = float(token.replace(",", "."))
    except ValueError:
        return None
    if value <= 0:
        return None
    return int(value) if value == int(value) else int(math.ceil(value))


def parse_numeric_meters(tags: dict[str, str], keys: Iterable[str]) -> float | None:
    for key in keys:
        raw = tags.get(key)
        if not raw:
            continue
        token = raw.strip().split(";")[0].strip().lower().replace(",", ".")
        for suffix in (" m", "m"):
            if token.endswith(suffix):
                token = token[: -len(suffix)].strip()
        try:
            value = float(token)
        except ValueError:
            continue
        if value >= 0:
            return value
    return None


def building_min_level(tags: dict[str, str]) -> int:
    for key in ("building:min_level", "min_level"):
        parsed = parse_level_value(tags.get(key))
        if parsed is not None:
            return max(0, parsed)
    return 0


def building_max_level(tags: dict[str, str]) -> int | None:
    for key in ("building:max_level", "max_level"):
        parsed = parse_level_value(tags.get(key))
        if parsed is not None:
            return parsed
    return None


def levels_from_tags(tags: dict[str, str]) -> int | None:
    for key in ("building:levels", "levels"):
        parsed = parse_level_value(tags.get(key))
        if parsed is not None:
            return parsed
    return None


def tag_has_value(tags: dict[str, str], key: str) -> bool:
    value = tags.get(key)
    return value is not None and value != ""


def has_floor_level_tags(tags: dict[str, str]) -> bool:
    if levels_from_tags(tags) is not None:
        return True
    max_level = building_max_level(tags)
    min_level = building_min_level(tags)
    return max_level is not None and max_level >= min_level


def height_from_meters_tag(tags: dict[str, str]) -> float | None:
    value = parse_numeric_meters(tags, ("height", "building:height"))
    if value is not None and value > 0:
        return min(value, MAX_BUILDING_LEVELS * FLOOR_HEIGHT)
    return None


def has_assigned_height(tags: dict[str, str]) -> bool:
    return has_floor_level_tags(tags) or height_from_meters_tag(tags) is not None


def part_base_offset(tags: dict[str, str]) -> float:
    min_height = parse_numeric_meters(tags, ("min_height", "building:min_height"))
    if min_height is not None:
        return max(0.0, min_height)
    return building_min_level(tags) * FLOOR_HEIGHT


def resolve_part_vertical(tags: dict[str, str]) -> tuple[float, float]:
    min_level = building_min_level(tags)
    max_level = building_max_level(tags)
    base_offset = part_base_offset(tags)
    levels_tag = levels_from_tags(tags)

    if max_level is not None and max_level >= min_level:
        level_count = min(max_level - min_level + 1, MAX_BUILDING_LEVELS)
    elif (
        levels_tag is not None
        and (tag_has_value(tags, "building:min_level") or tag_has_value(tags, "min_level"))
    ):
        level_count = min(
            levels_tag - min_level + 1 if levels_tag >= min_level else levels_tag,
            MAX_BUILDING_LEVELS,
        )
    elif levels_tag is not None:
        level_count = min(levels_tag, MAX_BUILDING_LEVELS)
    else:
        height_m = height_from_meters_tag(tags)
        if height_m is not None:
            return base_offset, height_m
        return base_offset, FLOOR_HEIGHT

    height = min(level_count * FLOOR_HEIGHT, MAX_BUILDING_LEVELS * FLOOR_HEIGHT)
    return base_offset, height


def resolve_main_building_vertical(tags: dict[str, str]) -> tuple[float, float]:
    if has_floor_level_tags(tags):
        max_level = building_max_level(tags)
        min_level = building_min_level(tags)
        if max_level is not None and max_level >= min_level:
            level_count = max_level - min_level + 1
        else:
            level_count = levels_from_tags(tags) or 1
        level_count = min(max(1, level_count), MAX_BUILDING_LEVELS)
        return 0.0, level_count * FLOOR_HEIGHT

    height_m = height_from_meters_tag(tags)
    if height_m is not None:
        return 0.0, height_m
    return 0.0, FLOOR_HEIGHT


def resolve_building_vertical(tags: dict[str, str], is_part: bool) -> tuple[float, float]:
    if is_part:
        return resolve_part_vertical(tags)
    return resolve_main_building_vertical(tags)


def close_ring(points: list[dict[str, float]]) -> list[tuple[float, float]] | None:
    if len(points) < 3:
        return None
    ring = [(point["lon"], point["lat"]) for point in points]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring if len(ring) >= 4 else None


def is_building_roof_tags(tags: dict[str, str] | None) -> bool:
    if not tags:
        return False
    if tags.get("building") == "roof":
        return True
    return tags.get("building:part") == "roof"


def is_building_part_tags(tags: dict[str, str] | None) -> bool:
    if not tags or is_building_roof_tags(tags):
        return False
    part = tags.get("building:part")
    return part not in (None, "", "no")


def is_building_way_tags(tags: dict[str, str] | None) -> bool:
    if not tags or is_building_roof_tags(tags):
        return False
    if is_building_part_tags(tags):
        return True
    return bool(tags.get("building"))


def part_member_tags(member: dict[str, Any], way_by_id: dict[int, dict[str, Any]]) -> dict[str, str]:
    way_tags = way_by_id.get(member["ref"], {}).get("tags") or {}
    inline_tags = member.get("tags") or {}
    return {**way_tags, **inline_tags}


def is_building_part_member(
    member: dict[str, Any],
    way_by_id: dict[int, dict[str, Any]],
) -> bool:
    tags = part_member_tags(member, way_by_id)
    if is_building_roof_tags(tags):
        return False
    if member.get("role") == "part":
        return True
    return is_building_part_tags(tags)


def relation_has_building_parts(relation: dict[str, Any], way_by_id: dict[int, dict[str, Any]]) -> bool:
    for member in relation.get("members") or []:
        if member.get("type") != "way" or not member.get("geometry"):
            continue
        if member.get("role") == "inner":
            continue
        if is_building_part_member(member, way_by_id):
            return True
    return False


def latlon_ring_to_polygon(
    ring: list[dict[str, float]],
    transformer: Transformer,
) -> Polygon | None:
    closed = close_ring(ring)
    if not closed:
        return None
    coords = [transformer.transform(lon, lat) for lon, lat in closed]
    polygon = Polygon(coords)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area < 1e-3:
        return None
    return polygon


def iter_buildings(elements: list[dict[str, Any]], transformer: Transformer) -> Iterable[BuildingFeature]:
    drawn: set[str] = set()
    way_by_id = {element["id"]: element for element in elements if element.get("type") == "way"}

    for element in elements:
        if element.get("type") != "way" or not element.get("geometry"):
            continue
        tags = element.get("tags") or {}
        if not is_building_way_tags(tags):
            continue
        osm_id = f"way/{element['id']}"
        if osm_id in drawn:
            continue
        polygon = latlon_ring_to_polygon(element["geometry"], transformer)
        if polygon is None:
            continue
        drawn.add(osm_id)
        yield BuildingFeature(
            osm_id=osm_id,
            tags=tags,
            geometry=polygon,
            inner_rings=[],
            is_part=is_building_part_tags(tags),
        )

    for element in elements:
        if element.get("type") != "relation" or not (element.get("tags") or {}).get("building"):
            continue
        if is_building_roof_tags(element.get("tags")):
            continue
        has_parts = relation_has_building_parts(element, way_by_id)
        inner_rings: list[Polygon] = []
        for member in element.get("members") or []:
            if member.get("type") != "way" or member.get("role") != "inner" or not member.get("geometry"):
                continue
            inner = latlon_ring_to_polygon(member["geometry"], transformer)
            if inner is not None:
                inner_rings.append(inner)

        for member in element.get("members") or []:
            if member.get("type") != "way" or not member.get("geometry"):
                continue
            if member.get("role") == "inner":
                continue
            role = member.get("role")
            if role and role not in ("outer", "part"):
                continue
            member_tags = part_member_tags(member, way_by_id)
            if is_building_roof_tags(member_tags):
                continue
            osm_id = f"way/{member['ref']}"
            if osm_id in drawn:
                continue
            polygon = latlon_ring_to_polygon(member["geometry"], transformer)
            if polygon is None:
                continue
            drawn.add(osm_id)
            is_part = is_building_part_member(member, way_by_id)
            if not is_part and has_parts:
                continue
            tags = member_tags if is_part else (element.get("tags") or {})
            yield BuildingFeature(
                osm_id=osm_id,
                tags=tags,
                geometry=polygon,
                inner_rings=[] if is_part else inner_rings,
                is_part=is_part,
            )


def features_to_gdf(features: list[BuildingFeature], layer: str) -> gpd.GeoDataFrame:
    records = []
    for feature in features:
        records.append(
            {
                "osm_id": feature.osm_id,
                "layer": layer,
                "is_part": feature.is_part,
                "tags_json": json.dumps(feature.tags, ensure_ascii=False, sort_keys=True),
                **feature.tags,
                "geometry": feature.geometry,
            }
        )
    if not records:
        return gpd.GeoDataFrame(columns=["osm_id", "layer", "is_part", "geometry"], geometry="geometry")
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:3857")


def spatial_join_parts_to_outlines(
    outlines: gpd.GeoDataFrame,
    parts: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if outlines.empty or parts.empty:
        return gpd.GeoDataFrame(
            columns=["part_osm_id", "outline_osm_id", "part_geometry", "outline_tags_json"],
            geometry="part_geometry",
        )

    parts_indexed = parts.copy()
    parts_indexed["part_osm_id"] = parts_indexed["osm_id"]
    parts_indexed = parts_indexed.rename(columns={"geometry": "part_geometry"})

    outlines_indexed = outlines.copy()
    outlines_indexed["outline_osm_id"] = outlines_indexed["osm_id"]
    outlines_indexed["outline_tags_json"] = outlines_indexed["tags_json"]
    outlines_indexed = outlines_indexed.rename(columns={"geometry": "outline_geometry"})

    joined = gpd.sjoin(
        parts_indexed,
        outlines_indexed[["outline_osm_id", "outline_geometry", "outline_tags_json"]],
        how="left",
        predicate="intersects",
    )

    if joined.empty:
        return joined

    joined["parent_area"] = joined["outline_geometry"].area
    joined = joined.sort_values(["part_osm_id", "parent_area"], ascending=[True, False])
    joined = joined.drop_duplicates(subset=["part_osm_id"], keep="first")
    return joined


def resolve_volume_tags(part_row: pd.Series | None, outline_tags: dict[str, str], part_tags: dict[str, str]) -> dict[str, str]:
    if has_assigned_height(part_tags):
        return part_tags
    return {**outline_tags, **part_tags}


def polygon_with_holes(outer: Polygon, holes: list[Polygon]) -> Polygon | MultiPolygon:
    if not holes:
        return outer
    hole_coords = [list(hole.exterior.coords) for hole in holes if not hole.is_empty]
    if not hole_coords:
        return outer
    shell = Polygon(outer.exterior.coords, holes=hole_coords)
    if not shell.is_valid:
        shell = shell.buffer(0)
    return shell


def subtract_part_footprints(outline: Polygon, part_geometries: list[Polygon]) -> list[Polygon]:
    if not part_geometries:
        return [outline]
    cutters = unary_union(part_geometries)
    if cutters.is_empty:
        return [outline]
    remainder = outline.difference(cutters)
    if remainder.is_empty:
        return []
    if isinstance(remainder, Polygon):
        return [remainder]
    return [geom for geom in remainder.geoms if isinstance(geom, Polygon) and not geom.is_empty]


def extrude_polygon_3d(polygon: Polygon, base_z: float, height: float) -> dict[str, Any]:
    if polygon.is_empty or height <= 0:
        return {"type": "Prism3D", "coordinates": []}

    bottom = [(x, y, base_z) for x, y in polygon.exterior.coords]
    top = [(x, y, base_z + height) for x, y in polygon.exterior.coords]
    walls = []
    for index in range(len(bottom) - 1):
        p0, p1 = bottom[index], bottom[index + 1]
        t0, t1 = top[index], top[index + 1]
        walls.append(
            {
                "type": "Polygon",
                "coordinates": [[p0, p1, t1, t0, p0]],
            }
        )

    return {
        "type": "Prism3D",
        "base_z": base_z,
        "top_z": base_z + height,
        "height_m": height,
        "footprint": mapping(polygon),
        "bottom": {"type": "Polygon", "coordinates": [bottom]},
        "top": {"type": "Polygon", "coordinates": [top]},
        "walls": walls,
    }


def build_volumes_3d(
    outlines: gpd.GeoDataFrame,
    parts: gpd.GeoDataFrame,
    joined: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    volume_rows: list[dict[str, Any]] = []

    outline_tags_by_id = {
        row.osm_id: json.loads(row.tags_json)
        for row in outlines.itertuples(index=False)
    }
    part_tags_by_id = {
        row.osm_id: json.loads(row.tags_json)
        for row in parts.itertuples(index=False)
    }

    joined_by_part = {}
    if not joined.empty:
        for row in joined.itertuples(index=False):
            joined_by_part[row.part_osm_id] = row

    for part_row in parts.itertuples(index=False):
        part_tags = part_tags_by_id[part_row.osm_id]
        join_row = joined_by_part.get(part_row.osm_id)
        outline_tags = {}
        if join_row is not None and join_row.outline_osm_id:
            outline_tags = outline_tags_by_id.get(join_row.outline_osm_id, {})
        effective_tags = resolve_volume_tags(join_row, outline_tags, part_tags)
        base_z, height = resolve_building_vertical(effective_tags, is_part=True)
        prism = extrude_polygon_3d(part_row.geometry, base_z, height)
        volume_rows.append(
            {
                "osm_id": part_row.osm_id,
                "source": "building:part",
                "parent_osm_id": join_row.outline_osm_id if join_row is not None else None,
                "base_z": base_z,
                "height_m": height,
                "top_z": base_z + height,
                "geometry": part_row.geometry,
                "geometry_3d": json.dumps(prism, ensure_ascii=False),
            }
        )

    for outline_row in outlines.itertuples(index=False):
        outline_tags = outline_tags_by_id[outline_row.osm_id]
        overlapping_parts = [
            part.geometry
            for part in parts.itertuples(index=False)
            if part.geometry.intersects(outline_row.geometry)
        ]
        remainders = subtract_part_footprints(outline_row.geometry, overlapping_parts)
        base_z, height = resolve_building_vertical(outline_tags, is_part=False)
        for index, footprint in enumerate(remainders):
            if footprint.area < 1.0:
                continue
            prism = extrude_polygon_3d(footprint, base_z, height)
            volume_rows.append(
                {
                    "osm_id": f"{outline_row.osm_id}#remainder-{index}",
                    "source": "building",
                    "parent_osm_id": outline_row.osm_id,
                    "base_z": base_z,
                    "height_m": height,
                    "top_z": base_z + height,
                    "geometry": footprint,
                    "geometry_3d": json.dumps(prism, ensure_ascii=False),
                }
            )

    if not volume_rows:
        return gpd.GeoDataFrame(columns=["osm_id", "source", "geometry"], geometry="geometry", crs=outlines.crs)

    return gpd.GeoDataFrame(volume_rows, geometry="geometry", crs=outlines.crs)


def load_osm_elements(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("elements") or []


def make_transformer(center_lat: float, center_lon: float) -> Transformer:
    proj = (
        f"+proj=tmerc +lat_0={center_lat} +lon_0={center_lon} "
        "+k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs("EPSG:4326", proj, always_xy=True)


def process_osm_json(elements: list[dict[str, Any]], center_lat: float, center_lon: float) -> gpd.GeoDataFrame:
    transformer = make_transformer(center_lat, center_lon)
    features = list(iter_buildings(elements, transformer))
    outlines = features_to_gdf([feature for feature in features if not feature.is_part], "building")
    parts = features_to_gdf([feature for feature in features if feature.is_part], "building:part")
    joined = spatial_join_parts_to_outlines(outlines, parts)
    return build_volumes_3d(outlines, parts, joined)


def main() -> int:
    parser = argparse.ArgumentParser(description="GeoPandas sjoin building outlines with building:part and build 3D volumes.")
    parser.add_argument("--input", required=True, help="Path to Overpass/OSM JSON file")
    parser.add_argument("--lat", type=float, required=True, help="Sector center latitude")
    parser.add_argument("--lon", type=float, required=True, help="Sector center longitude")
    parser.add_argument("--output", required=True, help="Output GeoJSON path for 2D footprints with 3D metadata")
    args = parser.parse_args()

    elements = load_osm_elements(args.input)
    volumes = process_osm_json(elements, args.lat, args.lon)
    volumes.to_file(args.output, driver="GeoJSON")
    print(f"Wrote {len(volumes)} 3D building volumes to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
