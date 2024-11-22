import logging
import numpy as np
from typing import Dict, Any, Optional, Union, Set, Tuple, List, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pickle
from collections import defaultdict
import gc
import json
import mmap
from tqdm import tqdm
import sqlite3
import os

@dataclass
class IDMappingMeta:
    """Stores metadata about ID mappings."""
    source_type: str
    target_type: str
    total_mappings: int
    creation_time: datetime
    batch_size: int
    description: Optional[str] = None

class IDMapper:
    """
    Manages ID mappings with SQLite backend for memory efficiency.
    """
    def __init__(
        self,
        cache_size: int = 10000,
        experiment_name: Optional[str] = None,
        batch_size: int = 1000,
        enable_gc: bool = True,
        db_path: Optional[str] = None
    ):
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.enable_gc = enable_gc
        self.experiment_name = experiment_name or f"id_mapper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.experiment_name}")
        self._setup_logging()
        
        # Initialize SQLite database with WAL mode for better concurrency
        self.db_path = db_path or f"data/mappings/{self.experiment_name}.db"
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_database()
        
        # Initialize in-memory cache with LRU strategy
        self.cache = {}
        self.cache_stats = defaultdict(int)
        
        # Track metadata
        self.mapping_meta = {}
        
        self.logger.info(
            f"Initialized IDMapper with cache_size={cache_size}, "
            f"batch_size={batch_size}"
        )

    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs/id_mapping")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        fh = logging.FileHandler(
            log_dir / f"{self.experiment_name}.log"
        )
        fh.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(fh)

    def _init_database(self):
        """Initialize SQLite database with optimized settings."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable WAL mode for better write performance
                conn.execute('PRAGMA journal_mode=WAL')
                conn.execute('PRAGMA synchronous=NORMAL')
                conn.execute('PRAGMA cache_size=-2000') # 2MB cache
                
                cursor = conn.cursor()
                
                # Create mappings table with optimized indices
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS mappings (
                        mapping_type TEXT,
                        source_idx INTEGER,
                        target_idx INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (mapping_type, source_idx)
                    ) WITHOUT ROWID
                """)
                
                # Create efficient index for reverse lookups
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_target_lookup 
                    ON mappings(mapping_type, target_idx, source_idx)
                """)
                
                # Create metadata table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS mapping_meta (
                        mapping_type TEXT PRIMARY KEY,
                        source_type TEXT,
                        target_type TEXT,
                        total_mappings INTEGER,
                        creation_time TIMESTAMP,
                        batch_size INTEGER,
                        description TEXT
                    ) WITHOUT ROWID
                """)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise

    def add_index_mapping(
        self,
        source_idx: int,
        target_idx: int,
        mapping_type: str
    ):
        """Add mapping with SQLite batch support."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Add mapping
                cursor.execute("""
                    INSERT OR REPLACE INTO mappings 
                    (mapping_type, source_idx, target_idx)
                    VALUES (?, ?, ?)
                """, (mapping_type, source_idx, target_idx))
                
                # Update cache if mapping type is cached
                if mapping_type in self.cache:
                    if len(self.cache[mapping_type]) >= self.cache_size:
                        # Remove oldest entry (LRU)
                        self.cache[mapping_type].pop(next(iter(self.cache[mapping_type])))
                    self.cache[mapping_type][source_idx] = target_idx
                
                # Update metadata
                if mapping_type not in self.mapping_meta:
                    self.mapping_meta[mapping_type] = IDMappingMeta(
                        source_type='index',
                        target_type='index',
                        total_mappings=1,
                        creation_time=datetime.now(),
                        batch_size=self.batch_size
                    )
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO mapping_meta
                        (mapping_type, source_type, target_type, total_mappings, 
                         creation_time, batch_size, description)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        mapping_type,
                        'index',
                        'index',
                        1,
                        datetime.now().isoformat(),
                        self.batch_size,
                        None
                    ))
                else:
                    cursor.execute("""
                        UPDATE mapping_meta 
                        SET total_mappings = total_mappings + 1
                        WHERE mapping_type = ?
                    """, (mapping_type,))
                    
                    self.mapping_meta[mapping_type].total_mappings += 1
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error adding index mapping: {e}")
            raise

    def _update_cache_from_batch(self, mappings: List[Tuple[int, int, str]]):
        """Update cache from batch data with LRU policy."""
        mapping_dict = defaultdict(dict)
        
        # Group mappings by type
        for src, tgt, m_type in mappings:
            mapping_dict[m_type][src] = tgt
        
        # Update cache for each mapping type
        for m_type, type_mappings in mapping_dict.items():
            if m_type in self.cache:
                # Apply LRU policy if cache is full
                available_space = self.cache_size - len(self.cache[m_type])
                if available_space < len(type_mappings):
                    # Remove oldest entries
                    remove_count = len(type_mappings) - available_space
                    for _ in range(remove_count):
                        self.cache[m_type].popitem(last=False)
                
                # Update cache
                self.cache[m_type].update(type_mappings)

    def add_mappings_batch(
        self,
        mappings: List[Tuple[int, int, str]],
        validate: bool = True
    ):
        """Add multiple mappings efficiently with validation."""
        try:
            # Validate mappings if requested
            if validate:
                invalid_mappings = [
                    m for m in mappings 
                    if not isinstance(m[0], int) or not isinstance(m[1], int)
                ]
                if invalid_mappings:
                    raise ValueError(f"Invalid mappings found: {invalid_mappings}")

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Begin transaction
                cursor.execute("BEGIN IMMEDIATE")
                
                try:
                    # Process in sub-batches for memory efficiency
                    for i in range(0, len(mappings), self.batch_size):
                        batch = mappings[i:i + self.batch_size]
                        
                        # Prepare batch data
                        batch_data = [
                            (m_type, src, tgt, datetime.now().isoformat())
                            for src, tgt, m_type in batch
                        ]
                        
                        # Insert batch
                        cursor.executemany("""
                            INSERT OR REPLACE INTO mappings 
                            (mapping_type, source_idx, target_idx, created_at)
                            VALUES (?, ?, ?, ?)
                        """, batch_data)
                        
                        # Update metadata
                        mapping_types = set(m[2] for m in batch)
                        for m_type in mapping_types:
                            count = sum(1 for _, _, mt in batch if mt == m_type)
                            
                            if m_type not in self.mapping_meta:
                                self.mapping_meta[m_type] = IDMappingMeta(
                                    source_type='index',
                                    target_type='index',
                                    total_mappings=count,
                                    creation_time=datetime.now(),
                                    batch_size=self.batch_size
                                )
                                
                                cursor.execute("""
                                    INSERT OR REPLACE INTO mapping_meta
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    m_type, 'index', 'index', count,
                                    datetime.now().isoformat(),
                                    self.batch_size, None
                                ))
                            else:
                                cursor.execute("""
                                    UPDATE mapping_meta 
                                    SET total_mappings = total_mappings + ?
                                    WHERE mapping_type = ?
                                """, (count, m_type))
                                
                                self.mapping_meta[m_type].total_mappings += count
                        
                        # Memory cleanup
                        if self.enable_gc and i % (self.batch_size * 10) == 0:
                            gc.collect()
                    
                    # Commit transaction
                    conn.commit()
                    
                except Exception as e:
                    conn.rollback()
                    raise e
                
                # Update cache if needed
                self._update_cache_from_batch(mappings)
                
        except Exception as e:
            self.logger.error(f"Error adding mappings batch: {e}")
            raise

    def get_index_mapping(
        self,
        source_idx: int,
        mapping_type: str,
        use_cache: bool = True
    ) -> Optional[int]:
        """Get mapped index with caching and database fallback."""
        try:
            # Check cache first
            if use_cache and mapping_type in self.cache:
                if source_idx in self.cache[mapping_type]:
                    self.cache_stats[f"{mapping_type}_hits"] += 1
                    return self.cache[mapping_type][source_idx]
                self.cache_stats[f"{mapping_type}_misses"] += 1
            
            # Query database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT target_idx FROM mappings
                    WHERE mapping_type = ? AND source_idx = ?
                """, (mapping_type, source_idx))
                
                result = cursor.fetchone()
                
                if result:
                    target_idx = result[0]
                    
                    # Update cache
                    if use_cache:
                        if mapping_type not in self.cache:
                            self.cache[mapping_type] = {}
                        if len(self.cache[mapping_type]) >= self.cache_size:
                            self.cache[mapping_type].pop(next(iter(self.cache[mapping_type])))
                        self.cache[mapping_type][source_idx] = target_idx
                    
                    return target_idx
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting index mapping: {e}")
            return None

    def get_mappings_batch(
        self,
        source_indices: List[int],
        mapping_type: str,
        use_cache: bool = True
    ) -> List[Optional[int]]:
        """Get multiple mappings efficiently with cache optimization."""
        try:
            results = []
            cache_hits = []
            db_queries = []
            
            # Check cache first
            if use_cache and mapping_type in self.cache:
                for idx in source_indices:
                    if idx in self.cache[mapping_type]:
                        cache_hits.append((idx, self.cache[mapping_type][idx]))
                        self.cache_stats[f"{mapping_type}_hits"] += 1
                    else:
                        db_queries.append(idx)
                        self.cache_stats[f"{mapping_type}_misses"] += 1
            else:
                db_queries = source_indices
            
            # Query database for missing indices
            db_results = []
            if db_queries:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Process in batches
                    for i in range(0, len(db_queries), self.batch_size):
                        batch = db_queries[i:i + self.batch_size]
                        placeholders = ','.join('?' * len(batch))
                        
                        cursor.execute(f"""
                            SELECT source_idx, target_idx 
                            FROM mappings
                            WHERE mapping_type = ? 
                            AND source_idx IN ({placeholders})
                        """, (mapping_type, *batch))
                        
                        db_results.extend(cursor.fetchall())
                        
                        # Update cache
                        if use_cache:
                            if mapping_type not in self.cache:
                                self.cache[mapping_type] = {}
                            for src, tgt in cursor.fetchall():
                                if len(self.cache[mapping_type]) >= self.cache_size:
                                    self.cache[mapping_type].popitem(last=False)
                                self.cache[mapping_type][src] = tgt
            
            # Combine results maintaining order
            mapping_dict = dict(cache_hits + db_results)
            results = [mapping_dict.get(idx) for idx in source_indices]
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting mappings batch: {e}")
            return [None] * len(source_indices)

    def _get_mapping_iterator(
        self,
        mapping_type: str,
        batch_size: Optional[int] = None
    ) -> Iterator[Tuple[int, int]]:
        """Memory-efficient iterator for mappings."""
        batch_size = batch_size or self.batch_size
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total count
                cursor.execute("""
                    SELECT COUNT(*) FROM mappings
                    WHERE mapping_type = ?
                """, (mapping_type,))
                total = cursor.fetchone()[0]
                
                # Yield batches
                for offset in range(0, total, batch_size):
                    cursor.execute("""
                        SELECT source_idx, target_idx FROM mappings
                        WHERE mapping_type = ?
                        ORDER BY source_idx
                        LIMIT ? OFFSET ?
                    """, (mapping_type, batch_size, offset))
                    
                    yield from cursor.fetchall()
                    
                    # Clear memory
                    if self.enable_gc and offset % (batch_size * 10) == 0:
                        gc.collect()
                        
        except Exception as e:
            self.logger.error(f"Error in mapping iterator: {e}")
            raise

    def get_mapping_stats(self) -> Dict[str, Any]:
        """Get comprehensive mapping statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get mapping counts
                cursor.execute("""
                    SELECT mapping_type, COUNT(*) 
                    FROM mappings 
                    GROUP BY mapping_type
                """)
                mapping_counts = dict(cursor.fetchall())
                
                # Get metadata
                cursor.execute("SELECT * FROM mapping_meta")
                metadata = {row[0]: dict(zip(
                    ['source_type', 'target_type', 'total_mappings', 
                     'creation_time', 'batch_size', 'description'],
                    row[1:]
                )) for row in cursor.fetchall()}
                
                stats = {
                    'mappings': mapping_counts,
                    'metadata': metadata,
                    'cache': {
                        'size': {
                            name: len(cache)
                            for name, cache in self.cache.items()
                        },
                        'stats': dict(self.cache_stats)
                    }
                }
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting mapping stats: {e}")
            return {}

    def save_mappings(self, output_file: Union[str, Path]):
        """Export mappings to file."""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Export mappings
                cursor.execute("SELECT * FROM mappings")
                mappings = cursor.fetchall()
                
                # Export metadata
                cursor.execute("SELECT * FROM mapping_meta")
                metadata = cursor.fetchall()
                
                # Prepare export data
                export_data = {
                    'mappings': mappings,
                    'metadata': metadata,
                    'cache_stats': dict(self.cache_stats),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Save to file
                with open(output_file, 'wb') as f:
                    pickle.dump(export_data, f)
                    
                self.logger.info(f"Saved mappings to {output_file}")
                
        except Exception as e:
            self.logger.error(f"Error saving mappings: {e}")
            raise

    def load_mappings(self, mappings_file: Union[str, Path]):
        """Load mappings from file with memory optimization."""
        try:
            # Clear existing database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM mappings")
                cursor.execute("DELETE FROM mapping_meta")
                conn.commit()
            
            # Load data in chunks
            with open(mappings_file, 'rb') as f:
                data = pickle.load(f)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert mappings in batches
                batch_size = self.batch_size
                mappings = data['mappings']
                
                for i in range(0, len(mappings), batch_size):
                    batch = mappings[i:i + batch_size]
                    cursor.executemany("""
                        INSERT INTO mappings 
                        (mapping_type, source_idx, target_idx, created_at)
                        VALUES (?, ?, ?, ?)
                    """, batch)
                    
                    # Clear memory after each batch
                    if self.enable_gc and i % (batch_size * 10) == 0:
                        gc.collect()
                
                # Insert metadata
                cursor.executemany("""
                    INSERT INTO mapping_meta
                    (mapping_type, source_type, target_type, total_mappings,
                     creation_time, batch_size, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, data['metadata'])
                
                conn.commit()
            
            # Update cache stats
            self.cache_stats.update(data.get('cache_stats', {}))
            
            # Clear and rebuild cache
            self.cache.clear()
            
            self.logger.info(f"Loaded mappings from {mappings_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading mappings: {e}")
            raise

    def clear_cache(self, mapping_type: Optional[str] = None):
        """Clear cache for specific or all mapping types."""
        if mapping_type:
            if mapping_type in self.cache:
                self.cache[mapping_type].clear()
                self.cache_stats[f"{mapping_type}_hits"] = 0
                self.cache_stats[f"{mapping_type}_misses"] = 0
        else:
            self.cache.clear()
            self.cache_stats.clear()
        
        if self.enable_gc:
            gc.collect()
            
        self.logger.info(
            f"Cleared cache for {mapping_type if mapping_type else 'all'} mapping types"
        )

    def get_reverse_mapping(
        self,
        target_idx: int,
        mapping_type: str
    ) -> List[int]:
        """Get source indices that map to a target index."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT source_idx FROM mappings
                    WHERE mapping_type = ? AND target_idx = ?
                """, (mapping_type, target_idx))
                
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"Error getting reverse mapping: {e}")
            return []

    def delete_mapping_type(self, mapping_type: str):
        """Delete all mappings of a specific type."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM mappings WHERE mapping_type = ?", (mapping_type,))
                cursor.execute("DELETE FROM mapping_meta WHERE mapping_type = ?", (mapping_type,))
                conn.commit()
            
            # Clear cache for this mapping type
            self.clear_cache(mapping_type)
            if mapping_type in self.mapping_meta:
                del self.mapping_meta[mapping_type]
            
            self.logger.info(f"Deleted all mappings for type: {mapping_type}")
            
        except Exception as e:
            self.logger.error(f"Error deleting mapping type: {e}")
            raise

    def vacuum_database(self):
        """Optimize database with improved error handling."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Disable WAL mode temporarily
                conn.execute('PRAGMA journal_mode=DELETE')
                # Vacuum database
                conn.execute("VACUUM")
                # Re-enable WAL mode
                conn.execute('PRAGMA journal_mode=WAL')
            self.logger.info("Database vacuumed successfully")
        except Exception as e:
            self.logger.error(f"Error vacuuming database: {e}")
            raise

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        try:
            self.vacuum_database()
        except Exception:
            pass

    def __del__(self):
        """Cleanup resources."""
        try:
            self.vacuum_database()
        except Exception:
            pass