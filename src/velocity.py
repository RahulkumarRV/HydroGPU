import tensorflow as tf

@tf.function  # Compile to a TensorFlow graph for performance
def compute_velocity_matrix(elevation_matrix, neighbor_indices, cell_size=1.0, k=1.0):
    """
    Compute velocity matrix using GPU acceleration and optimized memory handling.
    """
    elevation_matrix = tf.convert_to_tensor(elevation_matrix, dtype=tf.float32)
    neighbor_indices = tf.convert_to_tensor(neighbor_indices, dtype=tf.int32)
    rows, cols = tf.shape(elevation_matrix)[0], tf.shape(elevation_matrix)[1]
    
    # Convert neighbor indices to row and col positions
    neighbor_rows = neighbor_indices // cols
    neighbor_cols = neighbor_indices % cols
    
    # Get elevation values at neighbor positions
    indices = tf.stack([neighbor_rows, neighbor_cols], axis=-1)
    neighbor_elevation = tf.gather_nd(elevation_matrix, indices)
    
    # Compute elevation difference
    elevation_diff = elevation_matrix - neighbor_elevation
    
    # Compute Euclidean distance efficiently
    row_range = tf.range(rows, dtype=tf.float32)
    col_range = tf.range(cols, dtype=tf.float32)
    
    dx = tf.abs(tf.expand_dims(row_range, 1) - tf.cast(neighbor_rows, tf.float32)) * cell_size
    dy = tf.abs(tf.expand_dims(col_range, 0) - tf.cast(neighbor_cols, tf.float32)) * cell_size
    
    distance = tf.sqrt(dx**2 + dy**2 + 1e-6)  # Avoid division by zero
    
    # Compute slope and velocity
    slope = elevation_diff / distance
    velocity_matrix = k * tf.sqrt(tf.maximum(slope, 0))
    
    # Create mask for cells with elevation > 0
    valid_mask = elevation_matrix > 0
    
    # Set velocity to 0 for cells where elevation <= 0
    velocity_matrix = tf.where(valid_mask, velocity_matrix, tf.zeros_like(velocity_matrix))
    
    return velocity_matrix



