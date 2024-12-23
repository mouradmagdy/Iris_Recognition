## load database done

## Load trained model done

## select system mode ( enroll / Verify )

## for enrollment

    - take the image --> preprocessing --> resnet --> embedding --> save embedding in the database

## for verification

    - take the image --> preprocessing --> pairs(img , each pic in the database) --> siamese network --> match
    - if one match --> matched --> granted
    - more than one match --> a3la probabiliy fel matches --> granted
    - No matches --> access not grant
