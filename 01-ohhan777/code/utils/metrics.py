def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [1/3, 1/3, 1/3]  # weights for [building_iou, road_iou, flood_mean_iou] 
    return (x * w).sum(0)