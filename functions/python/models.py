from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, Mapping


class Model:
    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]):
        data = dict(mapping)
        if is_dataclass(cls):
            valid_fields = {field.name for field in fields(cls) if field.init}
            data = {key: value for key, value in data.items() if key in valid_fields}
        return cls(**data)

    def copy(self):
        return self.__class__(**self.to_mapping())

    def copy_with(self, **updates):
        data = self.to_mapping()
        data.update(updates)
        return self.__class__(**data)

    def to_mapping(self):
        if is_dataclass(self):
            return asdict(self)
        return dict(vars(self))


@dataclass
class Structure(Model):
    draft: Any = None
    M_Ballast: Any = None
    fb: Any = None
    rho_Steel: Any = None
    rho_Water: Any = None
    M_Tower: Any = None
    M_Turbine: Any = None
    z_CM_Tower: Any = None
    z_CM_Turbine: Any = None
    I_CM_Tower: Any = None
    Ballast_COG: Any = None
    CM: Any = None
    B11: Any = None
    K_Moor: Any = None
    z_Moor: Any = None
    ls: Any = None
    M: Any = None
    A: Any = None
    MA_inv: Any = None
    B: Any = None
    C: Any = None
    zCB: Any = None
    V_disp: Any = None
    buoyant_mass: Any = None
    fnat: Any = None
    MTot: Any = None
    zCM_Tot: Any = None
    IO_Tot: Any = None
    ms: Any = None
    mf: Any = None
    zCMs: Any = None
    zCMf: Any = None
    ICMs: Any = None
    ICMf: Any = None
    IAA: Any = None
    DProfile: Any = None
    ThicknessProfile: Any = None
    DMonopile: Any = None
    Thickness: Any = None
    z: Any = None
    zBeamNodal: Any = None
    phiNodal: Any = None
    zBeamElement: Any = None
    dz: Any = None
    phiElement: Any = None
    rhoAElement: Any = None
    GM: Any = None
    GD: Any = None
    GK: Any = None
    zhub: Any = None


@dataclass
class Response(Model):
    t: Any = None
    alpha: Any = None
    alphaDot: Any = None
    alphaDotDot: Any = None
    x1: Any = None
    x5: Any = None


@dataclass
class LoadSeries(Model):
    t: Any = None
    F: Any = None
    M: Any = None


@dataclass
class TimeInfo(Model):
    TDur: Any = None
    dt: Any = None
    TTrans: Any = None
    fHighCut: Any = None
    water_depth: Any = None
    turbulence_intensity: Any = None
    turbulence_length_scale: Any = None


@dataclass
class Constants(Model):
    g: Any = None
    rho_air: Any = None
    rho_water: Any = None