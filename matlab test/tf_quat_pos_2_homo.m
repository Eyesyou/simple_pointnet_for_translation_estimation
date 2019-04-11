function homopos = tf_quat_pos_2_homo(batch_pos)
batch = size(batch_pos,1)
w = batch_pos(:,1)
x = batch_pos(:,2)
y = batch_pos(:,3)
z = batch_pos(:,4)
trans_x = batch_pos(:,4)
trans_x = reshape(trans_x, 1, 1)
size(trans_x)

trans_y = batch_pos(:,5)
trans_z = batch_pos(:,6)




homopos=1
end

