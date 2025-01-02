
        # Load AuraSR model
        try:
            # First load the config
            config_path = "weights/config.json"
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
            with tqdm(total=3, desc="Loading model") as pbar:
                # Load config
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    pbar.update(1)
                    pbar.set_description("Config loaded, initializing model")
            
                # Create model instance
                self.model = AuraSR(config=config, device=self.device)
                pbar.update(1)
                pbar.set_description("Model initialized, loading weights")
            
                # Load weights
                if model_path.endswith('.safetensors'):
                    state_dict = load_file(model_path)
                else:
                    state_dict = torch.load(model_path)
                
                self.model.upsampler.load_state_dict(state_dict, strict=True)
                pbar.update(1)
                pbar.set_description("Model ready")

            print(f"Initialize: batch-size {batch_size} | is-overlap {overlap}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("\nFull stack trace:")
            traceback.print_exc()
            sys.exit(1)
        
        # Load restoration model
        try:
            if restoration_model in RESTORATION_MODELS:
                model_info = RESTORATION_MODELS[restoration_model]
                model_path = model_info['file']

                print(f"Loading restoration model: {model_path}")
                
                # Load state dict
                # if model_path.endswith('.safetensors'):
                state_dict = load_file(model_path)
                # else:
                #     state_dict = torch.load(model_path)
                    
                # Use spandrel to load the model
                self.restoration_model = ModelLoader(device=device).load_from_state_dict(state_dict)
                
                # Verify it's an image model
                if not isinstance(self.restoration_model, ImageModelDescriptor):
                    raise ValueError("Restoration model must be a single-image model")
                
                print(f"Successfully loaded restoration model with scale factor: {self.restoration_model.scale}")
                print(f"Model input channels: {self.restoration_model.input_channels}")
                print(f"Model output channels: {self.restoration_model.output_channels}")
            else:
                self.restoration_model = None
                print("No restoration model specified")
                
        except Exception as e:
            print(f"Error loading restoration model: {str(e)}")
            traceback.print_exc()
            sys.exit(1)


class VideoUpscaler:
    def __init__(
        self,
        restoration_model: str = "4xRealWebPhoto-v4-dat2",
        devices: Optional[str] = None,
        batch_size: int = 1,
        tile_size: int = 512,
        overlap: bool = True
    ):
        """Initialize the video upscaler."""
            # Parse and validate GPU devices
        self.devices = parse_gpu_devices(devices)
        
        # Validate model requirements
        if tile_size % 32 != 0 and tile_size >= 128:  # Assuming model requires multiple of 32
            raise ValueError("Tile size must be multiple of 32 and greater than or equal to 128")
        
        # Estimate memory requirements
        available_memory = torch.cuda.get_device_properties(0).total_memory
        # if self.estimate_batch_memory(batch_size, tile_size) > available_memory:
        #     raise ValueError("Batch size too large for available GPU memory")
        
        print(f"Available GPU memory {available_memory}")
        
        if not torch.cuda.is_available():
            raise FileNotFoundError(f"CUDA device is not available")
        
        # initialize local variables
        self.restoration_model = None
        self.batch_size = batch_size
        self.tile_size = tile_size
        self.overlap = overlap

        # initialize stats dictionary
        self.stats = {
            'input_file': None,
            'output_file': None,
            'start_time': None,
            'end_time': None,
            'total_duration': None,
            'metadata': {},
            'timings': {
                'extraction': None,
                'processing': None,
                'encoding': None
            },
            'processing_stats': {
                'total_frames': None,
                'processed_frames': 0,
                'failed_frames': [],
                'avg_frame_time': None,
                'gpu_memory_peak': None if torch.cuda.is_available() else 'N/A'
            }
        }

        # Load restoration model
        try:
            if restoration_model in RESTORATION_MODELS:
                model_info = RESTORATION_MODELS[restoration_model]
                model_path = model_info['file']

                print(f"Loading restoration model: {model_path}")
                
                # Load state dict
                state_dict = load_file(model_path)

                # Use spandrel to load the model
                self.restoration_model = ModelLoader(device=device).load_from_state_dict(state_dict)
                
                # Verify it's an image model
                if not isinstance(self.restoration_model, ImageModelDescriptor):
                    raise ValueError("Restoration model must be a single-image model")
                
                print(f"Successfully loaded restoration model with scale factor: {self.restoration_model.scale}")
                print(f"Model input channels: {self.restoration_model.input_channels}")
                print(f"Model output channels: {self.restoration_model.output_channels}")
            else:
                self.restoration_model = None
                print("No restoration model specified")
                
        except Exception as e:
            print(f"Error loading restoration model: {str(e)}")
            traceback.print_exc()
            sys.exit(1)
        

    def process_frames(self, input_dir: str, output_dir: str) -> None:
        """Process frames using restoration model with nested progress bars."""
        os.makedirs(output_dir, exist_ok=True)
        frames = sorted(os.listdir(input_dir))

        # Initialize tile parameters
        tile_size = self.tile_size
        overlap = 32
        min_tile_size = 128

        to_tensor = transforms.ToTensor()
        to_pil = transforms.ToPILImage()

        # Main progress bar for frames
        with tqdm(total=len(frames), desc="Processing frames") as pbar_frames:
            for i in range(0, len(frames), self.batch_size):
                batch_frames = frames[i:i + self.batch_size]
                try:
                    for frame in batch_frames:
                        input_path = os.path.join(input_dir, frame)
                        output_path = os.path.join(output_dir, frame)
                        
                        img = Image.open(input_path).convert('RGB')

                        if self.restoration_model is not None:
                            oom = True
                            current_tile = tile_size
                            img_tensor = to_tensor(img).unsqueeze(0).to(self.device)
                            
                            while oom and current_tile >= min_tile_size:
                                try:
                                    # Calculate number of tiles
                                    _, _, h, w = img_tensor.shape
                                    n_tiles_h = math.ceil(h / (current_tile - overlap))
                                    n_tiles_w = math.ceil(w / (current_tile - overlap))
                                    total_tiles = n_tiles_h * n_tiles_w
                                    
                                    # Create nested progress bar for tiles
                                    with tqdm(
                                        total=total_tiles, 
                                        desc=f"Tiling frame {frame} ({current_tile}px)", 
                                        leave=False
                                    ) as pbar_tiles:
                                        restored_tensor = tiled_scale(
                                            img_tensor,
                                            self.restoration_model,
                                            tile_x=current_tile,
                                            tile_y=current_tile,
                                            overlap=overlap,
                                            upscale_amount=self.restoration_model.scale,
                                            output_device=self.device,
                                            pbar=pbar_tiles
                                        )
                                    oom = False

                                except torch.cuda.OutOfMemoryError:
                                    current_tile //= 2
                                    torch.cuda.empty_cache()
                                    if current_tile < min_tile_size:
                                        raise RuntimeError("Unable to process frame even with minimum tile size")
                                    print(f"OOM error, reducing tile size to {current_tile}")

                            restored_tensor = torch.clamp(restored_tensor, 0, 1)
                            restored = to_pil(restored_tensor.squeeze().cpu())
                        else:
                            restored = img
                            
                        upscaled = restored

                        ###
                        # Potential second pass upscale and resize...
                        ###

                        upscaled.save(output_path, 'PNG')
                        pbar_frames.update(1)
                        
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    print("\nFull stack trace:")
                    traceback.print_exc()
                    continue
