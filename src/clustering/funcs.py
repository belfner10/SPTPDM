def get_all_sub(cluster_id, Z, num_rows):
    """
    Combines all sub clusters
    Parameters
    ----------
    cluster_id :
    Z :
        linkage matrix
    num_rows :

    Returns
    -------
    list: all data ids in the cluster
    """
    if cluster_id < num_rows:
        return [cluster_id]
    else:
        left_id = round(Z[cluster_id - num_rows][0])
        right_id = round(Z[cluster_id - num_rows][1])
        return get_all_sub(left_id, Z, num_rows) + get_all_sub(right_id, Z, num_rows)


# TODO does num_rows really need to be specified
def create_n_clusters(Z, num_rows, num_clusters=2):
    """
    creates the top 'num_clusters' clusters
    Parameters
    ----------
    Z :
        linkage matrix
    num_rows :
    num_clusters :

    Returns
    -------

    """
    cluster_seeds = [len(Z) + num_rows - 1]
    # incrementaly splits highest cluster on dendrogram until the desired number of clusters are found
    while len(cluster_seeds) < num_clusters:
        cid = max(cluster_seeds)
        cluster_seeds.remove(cid)
        cluster_seeds.append(round(Z[cid - num_rows][0]))
        cluster_seeds.append(round(Z[cid - num_rows][1]))

    ret = {}
    for x, cid in enumerate(cluster_seeds):
        for id in get_all_sub(cid, Z, num_rows):
            ret[id + 1] = x
    return ret
