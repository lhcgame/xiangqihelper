import os
import sys


def resource_dir():
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return resource_dir()


def candidate_paths(relative_path):
    if os.path.isabs(relative_path):
        return [relative_path]

    paths = []
    for base in (app_dir(), resource_dir()):
        path = os.path.abspath(os.path.join(base, relative_path))
        if path not in paths:
            paths.append(path)
    return paths


def resolve_resource_path(relative_path):
    for path in candidate_paths(relative_path):
        if os.path.exists(path):
            return path
    return candidate_paths(relative_path)[0]


def resolve_user_path(relative_path):
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.abspath(os.path.join(app_dir(), relative_path))
