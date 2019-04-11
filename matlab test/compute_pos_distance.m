function [angle_loss, trans_loss]=compute_pos_distance(batch_pos1, batch_pos2)
size(batch_pos1)
size(batch_pos2)
homo_pos1 = tf_quat_pos_2_homo(batch_pos1)


end



