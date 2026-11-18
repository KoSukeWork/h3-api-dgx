"""Server-owned recipes. Never accept arbitrary adapter paths or overrides."""

LARRY = "larry-turbo-v4"
LIGHTX = "lightx2v-ref2va-v0.1"


def resolve(request, settings):
    requested = request.profile
    if requested == "auto":
        if request.variant == "ref2va" and settings.ref2va_lora:
            requested = LIGHTX
        elif settings.turbo_lora:
            requested = LARRY
        else:
            requested = "base"
    if requested == LIGHTX:
        if not settings.ref2va_lora:
            raise ValueError("LightX2V Ref2VA profile is not enabled on this server")
        if request.task != "ref2va":
            raise ValueError("LightX2V Ref2VA requires task=ref2va, not text/first-last frame generation")
        if any(c.role != "reference" or c.type != "image" for c in request.conditions):
            raise ValueError("This Ref2VA profile currently accepts reference images only")
        if request.preset == "quality" or request.steps not in (None, 4):
            raise ValueError("LightX2V v0.1 requires 4 steps; use server_default, fast, or steps=4")
        return LIGHTX, {"mode": "turbo", "steps": 4, "num_inference_steps": 5}
    if requested == LARRY:
        if not settings.turbo_lora:
            raise ValueError("Larry Turbo profile is not enabled on this server")
        return LARRY, request.sampling(True)
    return "base", request.sampling(False)


def catalog(settings):
    items = []
    if settings.turbo_lora:
        items.append({"id": LARRY, "tasks": ["t2va", "fl2va"], "default_steps": 8,
                      "presets": {"fast": 4, "quality": 8}, "custom_steps_range": [1, 49],
                      "validated_steps": [4, 8], "quantized": False})
    if settings.ref2va_lora:
        items.append({"id": LIGHTX, "tasks": ["ref2va"], "default_steps": 4,
                      "presets": {"fast": 4}, "allowed_steps": [4], "quantized": False,
                      "reference_types": ["image"], "max_reference_images": 9,
                      "validated_reference_count": 1, "validated_short_edge": 480,
                      "flow_shift": 12.0, "audio_flow_shift": 3.0,
                      "reference_resize": "lightx2v-match-v1"})
    return items
