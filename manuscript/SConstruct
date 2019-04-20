from SCons.Script import VariantDir, Environment, Builder, Depends, Flatten
import os

VariantDir('_build', src_dir='.')

env = Environment(ENV=os.environ)
env['BUILDERS']['Latexdiff'] = Builder(action = 'latexdiff $SOURCES > $TARGET')

main, = env.PDF(target='_build/main.pdf',source='main.tex')
Default([main])

# When we have revised versions, we can use this to generate pdf diffs
#env.Latexdiff(target='diff.tex',source=['stored_main.tex','main.tex'])
#diff = env.PDF(target='diff.pdf',source='diff.tex')

Depends(Flatten([main]),
        Flatten(['refs.bib']))


cont_build = env.Command('.continuous', ['refs.bib', 'main.tex'],
    'while :; do inotifywait -e modify $SOURCES; scons -Q; done')
Alias('continuous', cont_build)
