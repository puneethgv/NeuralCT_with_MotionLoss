#!/usr/bin/env python
"""
XCAT Sinogram Adapter for NeuralCT Pipeline

This adapter bridges the XCAT cardiac simulation database with the NeuralCT
reconstruction pipeline, handling data format conversions and query operations.

Author: Ketan Vibhandik
Date: October 27, 2025
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import warnings

try:
    from cardiac_simulation.api import query_sinogram, load_sinogram
    from cardiac_simulation.config.lab_config import get_lab_config
    XCAT_AVAILABLE = True
except ImportError as e:
    XCAT_AVAILABLE = False
    IMPORT_ERROR = str(e)


class XCATSinogramAdapter:
    """
    Adapter class for loading XCAT simulation data into NeuralCT pipeline.
    
    Handles:
    - Querying XCAT registry with auto-generation
    - Loading sinogram data from registry
    - Format conversion (XCAT -> NeuralCT format)
    - Transpose handling ([n_angles, n_detectors] -> [n_detectors, n_angles])
    
    Example:
        >>> adapter = XCATSinogramAdapter(
        ...     patient_age=1,
        ...     sex="male",
        ...     heart_rate=60,
        ...     num_projections=1000,
        ...     frequency=4.0,
        ...     total_time=1.0,
        ...     detectors=128
        ... )
        >>> result = adapter.query_and_load()
        >>> sinogram = adapter.get_sinogram_data()
        >>> angles = adapter.get_angles()
    """
    
    def __init__(
        self,
        patient_age: int,
        sex: str,
        heart_rate: int,
        num_projections: int,
        frequency: float,
        total_time: float,
        detectors: int = 128
    ):
        """
        Initialize XCAT adapter with query parameters.
        
        Args:
            age: Patient age (e.g., 1 for male_1yr_ref, 5 for female_5yr_ref)
            sex: Patient sex (male or female)
            heart_rate: Heart rate in BPM
            num_projections: Total number of CT projections
            frequency: CT scanner temporal frequency in Hz
            total_time: Total scan duration in seconds
            detectors: Number of detectors (default: 128)
            
        Raises:
            ImportError: If cardiac_simulation package is not installed
        """
        if not XCAT_AVAILABLE:
            raise ImportError(
                f"cardiac_simulation package not available: {IMPORT_ERROR}\n"
                "Install with: pip install -e /path/to/simulation_code"
            )
        self.patient_age = patient_age
        self.sex = sex
        self.heart_rate = heart_rate
        self.num_projections = num_projections
        self.frequency = frequency
        self.total_time = total_time
        self.detectors = detectors
        
        # Query result storage
        self._query_result: Optional[Dict[str, Any]] = None
        self._sinogram_obj = None
        self._dataset_id: Optional[str] = None
        self.patient_name = f"{self.sex}_{self.patient_age}yr_ref"
        
        # XCAT configuration
        try:
            self._xcat_config = get_lab_config()
            self._xcat_base_path = Path(self._xcat_config.config.storage_base_path)
        except Exception as e:
            warnings.warn(f"Could not load XCAT lab config: {e}. Using default paths.")
            self._xcat_base_path = Path("/media/ExtraDrive1/kvibhandik/XCAT")
    
    def query_and_load(self) -> Dict[str, Any]:
        """
        Query XCAT registry and load sinogram data.
        
        Uses mode="generate" with auto_generate=True to automatically
        generate missing simulations.
        
        Returns:
            Dictionary with query result including:
                - status: "exact_match", "generated", "generating", "no_match"
                - dataset_id: Unique identifier for the simulation
                - sinograms: List of available sinograms
                - mode_used: Query mode that was used
                
        Raises:
            ValueError: If query fails or no data found
        """
        print("\n" + "="*60)
        print("XCAT SINOGRAM QUERY")
        print("="*60)
        print(f"Age:           {self.patient_age} years")
        print(f"Sex:           {self.sex}")
        print(f"Heart Rate:     {self.heart_rate} BPM")
        print(f"Projections:    {self.num_projections}")
        print(f"Frequency:      {self.frequency} Hz")
        print(f"Duration:       {self.total_time} sec")
        print(f"Detectors:      {self.detectors}")
        print("="*60)
        
        # Query with auto-generation enabled
        # Try generation first, fall back to nearest match if it fails
        try:
            self._query_result = query_sinogram(
                patient_name=self.patient_name,
                heart_rate=self.heart_rate,
                mode="generate",
                auto_generate=True,
                total_time=self.total_time,
                num_projections=self.num_projections,
                frequency=self.frequency,
                detectors=self.detectors
            )
        except (FileNotFoundError, RuntimeError) as e:
            # If generation fails (e.g., missing base.yml), fall back to nearest match
            warnings.warn(
                f"Auto-generation failed ({e}). Falling back to nearest match mode. "
                f"To enable generation, update to the latest xcat-simulation package."
            )
            self._query_result = query_sinogram(
                patient_name=self.patient_name,
                heart_rate=self.heart_rate,
                mode="nearest",
                auto_generate=False,
                detectors=self.detectors
            )
        
        status = self._query_result.get('status', 'unknown')
        
        # Handle different query statuses
        if status == 'exact_match':
            print(f"✓ Found existing simulation")
            self._dataset_id = self._query_result.get('dataset_id')
            if not self._dataset_id and 'sinograms' in self._query_result and self._query_result['sinograms']:
                self._dataset_id = self._query_result['sinograms'][0].get('dataset_id')
            print(f"  Dataset ID: {self._dataset_id}")
            
        elif status == 'nearest_match':
            print(f"✓ Found nearest matching simulation")
            self._dataset_id = self._query_result.get('dataset_id')
            if not self._dataset_id and 'sinograms' in self._query_result and self._query_result['sinograms']:
                self._dataset_id = self._query_result['sinograms'][0].get('dataset_id')
            print(f"  Dataset ID: {self._dataset_id}")
            if 'deltas' in self._query_result:
                deltas = self._query_result['deltas']
                if deltas.get('bpm_delta', 0) != 0:
                    print(f"  BPM delta: {deltas['bpm_delta']}")
            
        elif status == 'generated':
            print(f"✓ Generated new simulation")
            self._dataset_id = self._query_result['dataset_id']
            print(f"  Dataset ID: {self._dataset_id}")
            
        elif status == 'generating':
            print(f"⟳ Simulation in progress...")
            self._dataset_id = self._query_result.get('dataset_id')
            print(f"  Dataset ID: {self._dataset_id}")
            print(f"  This may take several minutes.")
            
        elif status == 'no_match':
            print(f"✗ No simulation found")
            raise ValueError(
                f"Could not find or generate simulation for {self.patient_name} "
                f"at {self.heart_rate} BPM"
            )
        elif status == 'generation_failed' or status == 'incomplete':
            print(f"⚠ Previous generation failed/incomplete")
            self._dataset_id = self._query_result.get('dataset_id')
            print(f"  Dataset ID: {self._dataset_id}")
            print(f"  Status: {status}")
            error_msg = self._query_result.get('error_message', 'No error details available')
            print(f"  Error: {error_msg}")
            
            # Check if the error is due to missing base.yml
            if 'base.yml' in str(error_msg):
                print(f"\n✗ ERROR: xcat-simulation package is missing required configuration files")
                print(f"  This version (1.0.0) does not support auto-generation.")
                print(f"\n  To fix this, update to the latest version from refactor2 branch:")
                print(f"    pip install --upgrade git+ssh://git@github.com/ucsd-fcrl/XCAT_Simulations.git@refactor2")
                print(f"\n  Or use an existing simulation by querying with mode='nearest'")
                raise RuntimeError(
                    "xcat-simulation package missing base.yml. "
                    "Update to refactor2 branch to enable auto-generation."
                )
            else:
                # For other failures, suggest manual cleanup
                print(f"\n  Failed dataset exists in database and blocks generation.")
                print(f"  To retry generation, manually delete the failed dataset:")
                print(f"    1. Connect to registry: /Data/data/registry/registry.db")
                print(f"    2. Delete dataset: {self._dataset_id}")
                print(f"    3. Rerun this script")
                raise ValueError(
                    f"Dataset {self._dataset_id} failed to generate. "
                    f"Delete it from the database and retry."
                )
        else:
            print(f"? Unknown status: {status}")
            raise ValueError(f"Unexpected query status: {status}")
        
        # Load the sinogram object
        print(f"\nLoading sinogram data...")
       
        # Get registry path from lab config
        # Instead of using load_sinogram(), query the database directly
        from cardiac_simulation.registry import SinogramQuery
        from cardiac_simulation.config.lab_config import get_lab_config
        
        lab_config = get_lab_config()
        registry_path = Path(lab_config.get_registry_db_path())
        
        query = SinogramQuery(registry_path)
        
        # Get the sinogram record (contains file_path)
        sinogram_record = query.get_main_sinogram(
            patient_name=self.patient_name,
            bpm=self.heart_rate
        )
        
        if not sinogram_record:
            raise ValueError(f"No sinogram found for {self.patient_name} @ {self.heart_rate} BPM")
        
        # Get the file path from the record
        file_path_host = sinogram_record["file_path"]
        
        # Translate to container path
        file_path_container = self._translate_path(file_path_host)
        
        # Load the file yourself using numpy
        import numpy as np
        data = np.load(file_path_container)
        
        # Create a simple object to hold the data
        from types import SimpleNamespace
        self._sinogram_obj = SimpleNamespace(
            sinogram=data['sinogram'],
            angles=data['angles']
        )
        
        print(f"✓ Loaded sinogram from: {file_path_container}")
        
        return self._query_result


    def _translate_path(self, host_path: str) -> Path:
        """
        Translate host path to container path.
        
        Maps: /media/ExtraDrive1/kvibhandik/XCAT/... -> /Data/...
        XCAT data will be in /Data/data/ as configured by xcat setup
        """
        host_prefix = "/media/ExtraDrive1/kvibhandik/XCAT"
        container_prefix = "/Data"
        
        if host_path.startswith(host_prefix):
            # Simple replacement: host path -> container path
            # XCAT setup configures storage at /Data with data in /Data/data/
            container_path = host_path.replace(host_prefix, container_prefix, 1)
            return Path(container_path)
        
        return Path(host_path)
    
    def get_sinogram_data(self) -> np.ndarray:
        """
        Get sinogram data in NeuralCT format.
        
        Converts from XCAT format [n_angles, n_detectors] to 
        NeuralCT format [n_detectors, n_angles] via transpose.
        
        Returns:
            Sinogram array in shape [n_detectors, n_angles]
            
        Raises:
            RuntimeError: If query_and_load() hasn't been called yet
        """
        if self._sinogram_obj is None:
            raise RuntimeError(
                "Must call query_and_load() before accessing sinogram data"
            )
        
        # XCAT format: [n_angles, n_detectors]
        # NeuralCT expects: [n_detectors, n_angles]
        # Therefore, we need to transpose
        sinogram_xcat = self._sinogram_obj.sinogram
        sinogram_neuralct = sinogram_xcat.T  # Transpose to [n_detectors, n_angles]
        
        return sinogram_neuralct
    
    def get_angles(self) -> np.ndarray:
        """
        Get projection angles in degrees.
        
        Returns:
            Array of angles in degrees, shape [n_angles]
            
        Raises:
            RuntimeError: If query_and_load() hasn't been called yet
        """
        if self._sinogram_obj is None:
            raise RuntimeError(
                "Must call query_and_load() before accessing angles"
            )
        
        return self._sinogram_obj.angles
    
    def get_dataset_id(self) -> str:
        """
        Get XCAT dataset identifier.
        
        Returns:
            Dataset ID string (e.g., "20250705_224445_abc123")
            
        Raises:
            RuntimeError: If query_and_load() hasn't been called yet
        """
        if self._dataset_id is None:
            raise RuntimeError(
                "Must call query_and_load() before accessing dataset ID"
            )
        
        return self._dataset_id
    
    def get_xcat_base_path(self) -> Path:
        """
        Get XCAT base path from configuration.
        
        Returns:
            Path to XCAT base directory (/Data)
        """
        return self._xcat_base_path
    
    def get_sinogram_path(self) -> Path:
        """
        Get full path to sinogram file in XCAT registry.
        
        XCAT structure: /Data/data/store/{dataset_id}/sinogram/
        (As configured by xcat setup wizard)
        
        Returns:
            Path to sinogram .npz file
            
        Raises:
            RuntimeError: If query_and_load() hasn't been called yet
        """
        if self._dataset_id is None:
            raise RuntimeError(
                "Must call query_and_load() before accessing sinogram path"
            )
        
        # XCAT registry structure: {base_path}/data/store/{dataset_id}/sinogram/
        # xcat_base_path is /Data, so full path is /Data/data/store/{dataset_id}/sinogram/
        sinogram_dir = (
            self._xcat_base_path / "data" / "store" / 
            self._dataset_id / "sinogram"
        )
        
        return sinogram_dir
    
    def validate_shape(self, expected_detectors: Optional[int] = None) -> bool:
        """
        Validate sinogram shape against expected dimensions.
        
        Args:
            expected_detectors: Expected number of detectors (optional)
            
        Returns:
            True if shape is valid
            
        Raises:
            RuntimeError: If query_and_load() hasn't been called yet
            ValueError: If shape validation fails
        """
        if self._sinogram_obj is None:
            raise RuntimeError(
                "Must call query_and_load() before validating shape"
            )
        
        sinogram = self.get_sinogram_data()
        angles = self.get_angles()
        
        n_detectors, n_angles = sinogram.shape
        
        # Validate number of angles matches
        if n_angles != len(angles):
            raise ValueError(
                f"Shape mismatch: sinogram has {n_angles} angle projections "
                f"but angles array has {len(angles)} elements"
            )
        
        # Validate number of projections matches request
        if len(angles) != self.num_projections:
            warnings.warn(
                f"Loaded {len(angles)} projections, requested {self.num_projections}. "
                f"This may be expected if downsampling occurred."
            )
        
        # Validate detector count if provided
        if expected_detectors is not None:
            if n_detectors != expected_detectors:
                raise ValueError(
                    f"Shape mismatch: sinogram has {n_detectors} detectors, "
                    f"expected {expected_detectors}"
                )
        
        print(f"✓ Shape validation passed: {sinogram.shape}")
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of loaded simulation.
        
        Returns:
            Dictionary with simulation metadata
            
        Raises:
            RuntimeError: If query_and_load() hasn't been called yet
        """
        if self._sinogram_obj is None or self._query_result is None:
            raise RuntimeError(
                "Must call query_and_load() before accessing summary"
            )
        
        sinogram = self.get_sinogram_data()
        angles = self.get_angles()
        
        return {
            'patient_name': self.patient_name,
            'heart_rate': self.heart_rate,
            'dataset_id': self._dataset_id,
            'query_status': self._query_result.get('status'),
            'sinogram_shape': sinogram.shape,
            'num_angles': len(angles),
            'angle_range': (angles.min(), angles.max()),
            'num_detectors': sinogram.shape[0],
            'xcat_base_path': str(self._xcat_base_path),
            'sinogram_path': str(self.get_sinogram_path())
        }
