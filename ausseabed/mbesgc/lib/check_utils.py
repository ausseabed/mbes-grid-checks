from typing import Type


def get_check(id, check_classes) -> Type | None:
    """
    Gets the check class for the given id. Will return None if check not found.

    Parameters:
        id (str UUID): id of the check to get
        check_classes (List of classes): List of all check classes

    """
    for cc in check_classes:
        if cc.id == id:
            return cc

    return None


def get_all_check_ids(check_classes):
    """
    Returns a list of the all check uuids that are supported by
    mbes-grid-checks.

    Parameters:
        check_classes (List of classes): List of all check classes

    Returns:
        list of strings: ids of all check classes (UUIDs)

    """
    return [check_class.id for check_class in check_classes]
