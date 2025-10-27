from setuptools import setup

setup(
    name='semldb',
    version='0.1.0',
    description='Semiconductor Machine Learning Models and Database',
    author='semldb',
    py_modules=['semldb'],
    packages=['models', 'models.CNTFET', 'models.NMOS', 'models.HFET', 'utils'],
    package_data={
        'models': [
            'CNTFET/*.pth',
            'NMOS/idvd_nmos_3_par_4_curves_linear/*.pth',
            'NMOS/idvd_nmos_3_par_4_curves_linear/*.pkl',
            'NMOS/idvg_nmos_3_par_2_curves_log_linear/*.pth',
            'NMOS/idvg_nmos_3_par_2_curves_log_linear/*.pkl',
            'HFET/IdVd/*.pth',
            'HFET/IdVg/*.pth',
            'HFET/IgVg/*.pth',
            'HFET/BV/*.pth',
        ],
    },
    install_requires=[
        'torch>=2.0.0',
        'numpy',
        'scipy',
        'pandas',
        'joblib',
        'scikit-learn>=1.0.0',
    ],
    python_requires='>=3.8',
)
