# # diffusify-engine/src/api/app.py
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from .routes import transformation, generation
# from ..diffusify_engine.transformation_manager import TransformationManager
# from ..diffusify_engine.pipelines.generative_pipeline import GenerativePipeline

# app = FastAPI(title="Diffusify Engine API")

# # Include routers for different functionalities
# app.include_router(transformation.router, prefix="/transform", tags=["transformation"])
# app.include_router(generation.router, prefix="/generate", tags=["generation"])

# # Example of a simple health check endpoint
# @app.get("/")
# async def root():
#     return {"message": "Diffusify Engine API is running"}