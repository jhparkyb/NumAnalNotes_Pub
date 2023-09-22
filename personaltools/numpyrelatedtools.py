
def check_np_dim(arr, arr_name="", suppress_arr=False):
    """
    This function prints dimension and shape of a numpy array
    Input:
        arr (ndarray): A numpy array whose dimension is inspected.
        arr_name (str): The name of the numpy array variable (for reporting purpose)
        suppress_arr (bool): (default=False) If this is True, the array itself is not printed. (appropriate when the array is large)
    Output:
        None
    Side effect:
        Print the dimension and the shape of the input array.
    """
    if arr_name == "":
        of_input_name = ""
    else:
        of_input_name = " of " + arr_name

    # construct dimension info
    dim_info = "dimension" + of_input_name + "= " + f"{arr.ndim}"

    # construct shape info
    shape_info = ", shape" + of_input_name + "= " + f"{arr.shape}"

    # construct final message
    if suppress_arr == True:
        arr_itself = ""
    else:
        arr_itself = arr

    print(arr_itself, dim_info, shape_info)

