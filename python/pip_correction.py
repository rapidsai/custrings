import os
import sys
import zipfile
import shutil
from hashlib import sha256
from base64 import urlsafe_b64encode


def convert_to_manylinux(name, version):
    """
    Modifies the arch metadata of a pip package linux_x86_64=>manylinux1_x86_64
    :param name:
    :param version:
    :return:
    """
    # Get python version as XY (27, 35, 36, etc)
    python_version = str(sys.version_info.major) + str(sys.version_info.minor)
    name_version = '{}-{}'.format(name.replace('-', '_'), version)

    # linux wheel package
    dist_zip = '{0}-cp{1}-cp{1}m-linux_x86_64.whl'.format(name_version,
                                                          python_version)
    dist_zip_path = os.path.join('dist', dist_zip)
    if not os.path.exists(dist_zip_path):
        print('Wheel not found: {}'.format(dist_zip_path))
        return

    unzip_dir = 'dist/unzip/{}'.format(dist_zip)
    os.makedirs(unzip_dir, exist_ok=True)
    with zipfile.ZipFile(dist_zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)

    wheel_file = '{}.dist-info/WHEEL'.format(name_version)
    new_wheel_str = ''
    with open(os.path.join(unzip_dir, wheel_file)) as f:
        for line in f.readlines():
            if line.startswith('Tag'):
                # Replace the linux tag
                new_wheel_str += line.replace('linux', 'manylinux1')
            else:
                new_wheel_str += line

    # compute hash & size of the new WHEEL file
    # Follows https://www.python.org/dev/peps/pep-0376/#record
    m = sha256()
    m.update(new_wheel_str.encode('utf-8'))
    hash = urlsafe_b64encode(m.digest()).decode('utf-8')
    hash = hash.replace('=', '')

    with open(os.path.join(unzip_dir, wheel_file), 'w') as f:
        f.write(new_wheel_str)
    statinfo = os.stat(os.path.join(unzip_dir, wheel_file))
    byte_size = statinfo.st_size

    record_file = os.path.join(unzip_dir,
                               '{}.dist-info/RECORD'.format(name_version))
    new_record_str = ''
    with open(record_file) as f:
        for line in f.readlines():
            if line.startswith(wheel_file):
                # Update the record for the WHEEL file
                new_record_str += '{},sha256={},{}'.format(wheel_file, hash,
                                                           str(byte_size))
                new_record_str += os.linesep
            else:
                new_record_str += line

    with open(record_file, 'w') as f:
        f.write(new_record_str)

    def zipdir(path, ziph):
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file),
                           os.path.join(root, file).replace(path, ''))

    new_zip_name = dist_zip.replace('linux', 'manylinux1')
    print('Generating new zip {}...'.format(new_zip_name))
    zipf = zipfile.ZipFile(os.path.join('dist', new_zip_name),
                           'w', zipfile.ZIP_DEFLATED)
    zipdir(unzip_dir, zipf)
    zipf.close()

    shutil.rmtree(unzip_dir, ignore_errors=True)
    os.remove(dist_zip_path)
