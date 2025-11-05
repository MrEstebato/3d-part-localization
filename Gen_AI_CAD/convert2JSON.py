import cadquery as cq

step_model = cq.importers.importStep("CARPETS_EXERCISE-1_CARPET.STEP")
cq.exporters.export(step_model, "Carpet1.json", "TJS")
