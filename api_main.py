from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.controllers.system_controller import SystemController
from utils.logger.logger import Logger
import pandas as pd
import io
import mimetypes
import tempfile
import os

app = FastAPI()

# Initialize Logger and SystemController
Logger.initialize()
Logger.disable_logging() # Disable logging for API by default, can be configured via an endpoint if needed
controller = SystemController()

class ExportRequest(BaseModel):
    data_format: str
    image_format: str
    path: str

class UpdateParametersRequest(BaseModel):
    params: dict

@app.get("/", tags=["Root"])
async def read_root():
    Logger.log("Root endpoint accessed")
    return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "FibriNet API is running!"})

@app.post("/input_network/", tags=["Network Operations"])
async def input_network(file: UploadFile = File(...)):
    Logger.log(f"Input network endpoint accessed with file: {file.filename}")
    import mimetypes
    if file.content_type not in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only .xlsx and .xls are supported.")
    
    try:
        file_content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        try:
            controller.input_network(tmp_path)
            return JSONResponse(status_code=status.HTTP_200_OK, content={"message": f"Network loaded successfully from {file.filename}"})
        finally:
            os.remove(tmp_path) # Clean up the temporary file
    except Exception as e:
        Logger.log(f"Error loading network: {e}", Logger.LogPriority.ERROR)
        raise HTTPException(status_code=500, detail=f"Failed to load network: {e}")

@app.post("/degrade_node/{node_id}", tags=["Network Operations"])
async def degrade_node(node_id: str):
    Logger.log(f"Degrade node endpoint accessed for node: {node_id}")
    try:
        controller.degrade_node(node_id)
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": f"Node {node_id} degraded successfully"})
    except Exception as e:
        Logger.log(f"Error degrading node {node_id}: {e}", Logger.LogPriority.ERROR)
        raise HTTPException(status_code=500, detail=f"Failed to degrade node {node_id}: {e}")

@app.post("/degrade_edge/{edge_id}", tags=["Network Operations"])
async def degrade_edge(edge_id: str):
    Logger.log(f"Degrade edge endpoint accessed for edge: {edge_id}")
    try:
        controller.degrade_edge(edge_id)
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": f"Edge {edge_id} degraded successfully"})
    except Exception as e:
        Logger.log(f"Error degrading edge {edge_id}: {e}", Logger.LogPriority.ERROR)
        raise HTTPException(status_code=500, detail=f"Failed to degrade edge {edge_id}: {e}")

@app.post("/undo_degradation/", tags=["Network Operations"])
async def undo_degradation():
    Logger.log("Undo degradation endpoint accessed")
    try:
        controller.undo_degradation()
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Last degradation undone successfully"})
    except Exception as e:
        Logger.log(f"Error undoing degradation: {e}", Logger.LogPriority.ERROR)
        raise HTTPException(status_code=500, detail=f"Failed to undo degradation: {e}")

@app.post("/redo_degradation/", tags=["Network Operations"])
async def redo_degradation():
    Logger.log("Redo degradation endpoint accessed")
    try:
        controller.redo_degradation()
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Last undone degradation redone successfully"})
    except Exception as e:
        Logger.log(f"Error redoing degradation: {e}", Logger.LogPriority.ERROR)
        raise HTTPException(status_code=500, detail=f"Failed to redo degradation: {e}")

@app.post("/run_simulation/", tags=["Simulation"])
async def run_simulation():
    Logger.log("Run simulation endpoint accessed")
    try:
        controller.run_simulation()
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Simulation ran successfully"})
    except Exception as e:
        Logger.log(f"Error running simulation: {e}", Logger.LogPriority.ERROR)
        raise HTTPException(status_code=500, detail=f"Failed to run simulation: {e}")

@app.get("/analyze_network/", tags=["Analysis"])
async def analyze_network():
    Logger.log("Analyze network endpoint accessed")
    try:
        analysis_results = controller.analyze_network()
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Network analysis complete", "results": analysis_results})
    except Exception as e:
        Logger.log(f"Error analyzing network: {e}", Logger.LogPriority.ERROR)
        raise HTTPException(status_code=500, detail=f"Failed to analyze network: {e}")

@app.post("/export_data/", tags=["Export"])
async def export_data(request: ExportRequest):
    Logger.log(f"Export data endpoint accessed with request: {request.dict()}")
    try:
        controller.export_data(request.dict())
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Data exported successfully"})
    except Exception as e:
        Logger.log(f"Error exporting data: {e}", Logger.LogPriority.ERROR)
        raise HTTPException(status_code=500, detail=f"Failed to export data: {e}")

@app.post("/update_parameters/", tags=["Configuration"])
async def update_parameters(request: UpdateParametersRequest):
    Logger.log(f"Update parameters endpoint accessed with params: {request.params}")
    try:
        controller.update_parameters(request.params)
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Parameters updated successfully"})
    except Exception as e:
        Logger.log(f"Error updating parameters: {e}", Logger.LogPriority.ERROR)
        raise HTTPException(status_code=500, detail=f"Failed to update parameters: {e}")
